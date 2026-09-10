const conversation = [];
let dialogueState = null;

const chatForm = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const conversationElement = document.querySelector("#conversation");
const courseTemplate = document.querySelector("#course-card-template");
const resetButton = document.querySelector("#reset-chat");
const welcomeMessage = conversationElement.firstElementChild.cloneNode(true);
let activeRequestController = null;
let requestVersion = 0;

function scrollToLatest() {
  conversationElement.scrollTop = conversationElement.scrollHeight;
}

function appendInlineMarkdown(container, text) {
  const tokenPattern = /(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|__([^_]+)__|\*([^*]+)\*|_([^_]+)_)/g;
  let lastIndex = 0;

  for (const match of text.matchAll(tokenPattern)) {
    container.append(document.createTextNode(text.slice(lastIndex, match.index)));
    let element;
    if (match[2]) {
      element = document.createElement("a");
      element.href = match[3];
      element.target = "_blank";
      element.rel = "noopener noreferrer";
      element.textContent = match[2];
    } else if (match[4]) {
      element = document.createElement("code");
      element.textContent = match[4];
    } else if (match[5] || match[6]) {
      element = document.createElement("strong");
      element.textContent = match[5] || match[6];
    } else {
      element = document.createElement("em");
      element.textContent = match[7] || match[8];
    }
    container.append(element);
    lastIndex = match.index + match[0].length;
  }
  container.append(document.createTextNode(text.slice(lastIndex)));
}

function formatMarkdown(text) {
  const fragment = document.createDocumentFragment();
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  let list = null;

  const closeList = () => { list = null; };
  lines.forEach((line) => {
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const unordered = line.match(/^[-*+]\s+(.+)$/);
    const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    if (heading) {
      closeList();
      const element = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(element, heading[2]);
      fragment.append(element);
    } else if (unordered || ordered) {
      const type = ordered ? "ol" : "ul";
      if (!list || list.tagName.toLowerCase() !== type) {
        list = document.createElement(type);
        fragment.append(list);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, (unordered || ordered)[1]);
      list.append(item);
    } else if (line.trim()) {
      closeList();
      const paragraph = document.createElement("p");
      appendInlineMarkdown(paragraph, line);
      fragment.append(paragraph);
    } else {
      closeList();
    }
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
    if (role === "assistant") {
      bubble.append(formatMarkdown(content));
    } else {
      bubble.textContent = content;
    }
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
    addCourseMetadata(meta, "", course.price !== undefined && course.price !== null ? `฿${course.price}` : null, "◇");
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
  const version = ++requestVersion;
  activeRequestController = new AbortController();
  const priorConversation = [...conversation];
  conversation.push({ role: "user", content: query });
  createMessage("user", query);
  const loadingMessage = createMessage("assistant", "", { loading: true });
  setPending(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, conversation: priorConversation, dialogue_state: dialogueState }),
      signal: activeRequestController.signal,
    });

    if (!response.ok) throw new Error("The server could not process the request.");
    const data = await response.json();
    if (!data || typeof data.answer !== "string") throw new Error("The server returned an incomplete response.");
    if (version !== requestVersion) return;

    loadingMessage.remove();
    const assistantMessage = createMessage("assistant", data.answer);
    addRelatedCourses(assistantMessage, data.related_courses);
    conversation.push({ role: "assistant", content: data.answer });
    if (data.dialogue_state && typeof data.dialogue_state === "object") {
      dialogueState = data.dialogue_state;
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    loadingMessage.remove();
    createMessage("assistant", "Unable to process your request. Please try again.", { error: true });
    console.error("Chat request failed:", error);
  } finally {
    if (version !== requestVersion) return;
    activeRequestController = null;
    setPending(false);
    messageInput.focus();
    scrollToLatest();
  }
}

function resetConversation() {
  requestVersion += 1;
  activeRequestController?.abort();
  activeRequestController = null;
  conversation.length = 0;
  dialogueState = null;
  conversationElement.replaceChildren(welcomeMessage.cloneNode(true));
  setPending(false);
  messageInput.value = "";
  resizeInput();
  messageInput.focus();
  scrollToLatest();
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
resetButton.addEventListener("click", resetConversation);
resizeInput();
