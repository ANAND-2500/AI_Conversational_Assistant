import { useState } from "react";

function Sidebar({
  chats,
  activeId,
  onSelect,
  onNewChat,
  onRename,
  onDelete
}) {
  const [editingId, setEditingId] = useState(null);
  const [tempTitle, setTempTitle] = useState("");

  function startEdit(chat, e) {
    e.stopPropagation();
    setEditingId(chat.id);
    setTempTitle(chat.title);
  }

  function saveEdit(chatId) {
    const title = tempTitle.trim();
    if (title) {
      onRename(chatId, title);
    }
    setEditingId(null);
  }

  function cancelEdit() {
    setEditingId(null);
    setTempTitle("");
  }

  return (
    <aside className="sidebar">
      <button className="new-chat" onClick={onNewChat}>
        + New Chat
      </button>

      <div className="chat-list">
        {Object.values(chats).map(chat => (
          <div
            key={chat.id}
            className={`chat-item ${
              chat.id === activeId ? "active" : ""
            }`}
            onClick={() => onSelect(chat.id)}
          >
            {editingId === chat.id ? (
              <input
                className="rename-input"
                value={tempTitle}
                autoFocus
                onChange={e => setTempTitle(e.target.value)}
                onBlur={() => saveEdit(chat.id)}
                onKeyDown={e => {
                  if (e.key === "Enter") saveEdit(chat.id);
                  if (e.key === "Escape") cancelEdit();
                }}
                onClick={e => e.stopPropagation()}
              />
            ) : (
              <>
                {/* ✅ FIXED: onClick is a PROP, not text */}
                <span
                  className="chat-title"
                  onClick={() => onSelect(chat.id)}
                >
                  {chat.title}
                </span>

                <div
                  className="chat-actions"
                  onClick={e => e.stopPropagation()}
                >
                  <button
                    title="Rename"
                    onClick={e => startEdit(chat, e)}
                  >
                    ✏️
                  </button>
                  <button
                    title="Delete"
                    onClick={() => onDelete(chat.id)}
                  >
                    🗑️
                  </button>
                </div>
              </>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}

export default Sidebar;
