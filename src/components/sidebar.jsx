import { useState } from "react";
import { Pencil, Trash2 } from "lucide-react";


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
    if (title) onRename(chatId, title);
    setEditingId(null);
  }

  function cancelEdit() {
    setEditingId(null);
    setTempTitle("");
  }

  return (
    <aside className="sidebar">
      <button className="new-chat" onClick={onNewChat}>+ New Chat</button>

      <div className="chat-list">
        {Object.values(chats).map(chat => (
          <div
            key={chat.id}
            className={`chat-item ${chat.id === activeId ? "active" : ""}`}
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
                <span className="chat-title">{chat.title}</span>

                <div className="chat-actions" onClick={e => e.stopPropagation()}>
                  <button className="icon-btn" onClick={e => startEdit(chat, e)}> <Pencil size={16} /> </button>
                  <button className="icon-btn" onClick={() => onDelete(chat.id)}> <Trash2 size={16} /> </button>
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
