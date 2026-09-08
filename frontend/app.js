const conversation = [];

const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const conversationElement = document.querySelector("#conversation");
const courseTemplate = document.querySelector("#course-card-template");

function scrollToLatest() {
  conversationElement.scrollTop = conversationElement.scrollHeight;
}

function formatText(text) {
  const fragment = document.createDocumentFragment();
  String(text || "").split("\n").forEach((line, index) => {
    if (index) fragment.append(document.createElement("br"));
    fragment.append(document.createTextNode(line));
  });
  return fragment;
}

function createMessage(role, content, options = {}) {
  const row = document.createElement("article");
  row.className = `message-row ${role}-message`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar bot-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "🤖";
    row.append(avatar);
  }

  const contentWrapper = document.createElement("div");
  contentWrapper.className = "message-content";
  const bubble = document.createElement("div");
  bubble.className = `bubble${options.error ? " error-bubble" : ""}${options.loading ? " loading-bubble" : ""}`;

  if (options.loading) {
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span><span class="sr-only">Thinking…</span>';
  } else {
    bubble.append(formatText(content));
  }

  contentWrapper.append(bubble);
  row.append(contentWrapper);

  if (role === "user") {
    const avatar = document.createElement("div");
    avatar.className = "avatar user-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "●";
    row.append(avatar);
  }

  conversationElement.append(row);
  scrollToLatest();
  return row;
}

function addCourseMetadata(container, label, value, icon) {
  if (value === undefined || value === null || value === "") return;
  const item = document.createElement("span");
  item.className = "meta-item";
  item.textContent = `${icon} ${label ? `${label}: ` : ""}${value}`;
  container.append(item);
}

function addRelatedCourses(message, courses) {
  if (!Array.isArray(courses) || courses.length === 0) return;

  const section = document.createElement("section");
  section.className = "related-courses";
  section.innerHTML = "<h2>▰ Related Courses</h2>";
  const grid = document.createElement("div");
  grid.className = "course-grid";

  courses.forEach((course, index) => {
    const card = courseTemplate.content.cloneNode(true);
    const level = card.querySelector(".level-badge");
    const symbol = card.querySelector(".course-symbol");
    const id = course.course_id || "Course";

    card.querySelector(".course-id").textContent = id;
    card.querySelector(".course-name").textContent = course.course_name || "Course information";
    card.querySelector(".course-description").textContent = course.description || "";
    symbol.textContent = ["✦", "⌁", "◈", "✺"][index % 4];

    if (course.level) {
      level.textContent = course.level;
      level.classList.add(`level-${String(course.level).toLowerCase().replace(/[^a-z]/g, "")}`);
    } else {
      level.remove();
    }

    const meta = card.querySelector(".course-meta");
    addCourseMetadata(meta, "", course.instructor, "♙");
    addCourseMetadata(meta, "", course.duration, "◷");
    addCourseMetadata(meta, "", course.schedule, "▣");
    addCourseMetadata(meta, "", course.price !== undefined && course.price !== null ? `$${course.price}` : null, "◇");
    if (!meta.childElementCount) meta.remove();

    grid.append(card);
  });

  section.append(grid);
  message.querySelector(".message-content").append(section);
  scrollToLatest();
}

function setPending(isPending) {
  messageInput.disabled = isPending;
  sendButton.disabled = isPending;
  chatForm.classList.toggle("is-pending", isPending);
}

function resizeInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 132)}px`;
}

async function sendMessage(query) {
  const priorConversation = [...conversation];
  conversation.push({ role: "user", content: query });
  createMessage("user", query);
  const loadingMessage = createMessage("assistant", "", { loading: true });
  setPending(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, conversation: priorConversation }),
    });

    if (!response.ok) throw new Error("The server could not process the request.");
    const data = await response.json();
    if (!data || typeof data.answer !== "string") throw new Error("The server returned an incomplete response.");

    loadingMessage.remove();
    const assistantMessage = createMessage("assistant", data.answer);
    addRelatedCourses(assistantMessage, data.related_courses);
    conversation.push({ role: "assistant", content: data.answer });
  } catch (error) {
    loadingMessage.remove();
    createMessage("assistant", "Unable to process your request. Please try again.", { error: true });
    console.error("Chat request failed:", error);
  } finally {
    setPending(false);
    messageInput.focus();
    scrollToLatest();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = messageInput.value.trim();
  if (!query || sendButton.disabled) return;
  messageInput.value = "";
  resizeInput();
  sendMessage(query);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

messageInput.addEventListener("input", resizeInput);
resizeInput();
