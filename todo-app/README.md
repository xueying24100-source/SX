# ✅ To-Do List Application

A fully functional, modern To-Do List web application built with **vanilla JavaScript (ES6+)**, featuring local storage persistence, dark/light mode, drag-and-drop reordering, and due dates.

---

## 🚀 How to Run

No build step required. Simply open `index.html` in any modern browser:

```bash
# Option 1 — Double-click index.html in your file manager

# Option 2 — Serve locally with Python
python3 -m http.server 8080
# Then open http://localhost:8080/todo-app/

# Option 3 — Serve with Node.js (npx)
npx serve .
```

---

## ✨ Features

- [x] **Add Tasks** — Input field + button to add new to-do items
- [x] **Complete Tasks** — Checkbox to mark tasks as done (visual strikethrough)
- [x] **Delete Tasks** — Remove individual tasks with animated exit
- [x] **Edit Tasks** — Inline editing via edit button or double-click on task text
- [x] **Filter Tasks** — Filter by: All / Active / Completed
- [x] **Clear Completed** — Bulk-delete all completed tasks
- [x] **Task Counter** — Shows remaining active task count
- [x] **Local Storage** — Tasks and theme persist across page refreshes
- [x] **Drag to Reorder** — HTML5 drag-and-drop to change task order
- [x] **Due Dates** — Optional due date per task with overdue highlighting
- [x] **Dark / Light Mode** — Manual toggle + respects `prefers-color-scheme`
- [x] **Responsive** — Works on mobile, tablet, and desktop
- [x] **Accessible** — ARIA labels, keyboard navigation, `aria-live` regions

---

## 📁 File Structure

```
todo-app/
├── index.html          ← Main HTML entry point (semantic HTML5)
├── css/
│   └── style.css       ← All styles (CSS variables, themes, animations, responsive)
├── js/
│   ├── app.js          ← TodoApp controller — connects Model ↔ View ↔ Storage
│   ├── storage.js      ← StorageManager class — localStorage abstraction
│   ├── ui.js           ← UIManager class — DOM rendering & event binding
│   └── drag.js         ← DragManager class — HTML5 drag-and-drop reordering
└── README.md           ← This file
```

---

## 🏗️ Architecture (MVC-like)

| Layer | File | Responsibility |
|-------|------|----------------|
| **Controller** | `js/app.js` | Orchestrates app lifecycle, handles user events |
| **View** | `js/ui.js` | Renders DOM, emits user-action callbacks |
| **Model/Storage** | `js/storage.js` | Reads/writes `localStorage`, JSON serialization |
| **Feature** | `js/drag.js` | Encapsulated drag-and-drop logic |

---

## 💾 localStorage Schema

All data is stored under two keys:

### `todoApp_tasks` — Array of task objects

```json
[
  {
    "id": "task-1710000000000-a1b2c",
    "text": "Buy groceries",
    "completed": false,
    "createdAt": "2024-03-10T09:00:00.000Z",
    "dueDate": "2024-03-12"
  },
  {
    "id": "task-1710000001000-d3e4f",
    "text": "Read a book",
    "completed": true,
    "createdAt": "2024-03-10T10:30:00.000Z",
    "dueDate": null
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique identifier (`task-{timestamp}-{random}`) |
| `text` | `string` | Task description (max 200 chars) |
| `completed` | `boolean` | Whether the task is done |
| `createdAt` | `string` | ISO 8601 creation timestamp |
| `dueDate` | `string \| null` | Optional due date (`YYYY-MM-DD`) or `null` |

### `todoApp_theme` — Theme preference

```
"light"  or  "dark"
```

---

## 🛠️ Technical Highlights

- **Vanilla JS only** — zero dependencies, no build tools
- **ES6+ syntax** — classes, arrow functions, destructuring, spread, template literals
- **XSS prevention** — all user input is HTML-escaped before rendering
- **Storage error handling** — catches `QuotaExceededError` gracefully
- **JSDoc comments** — all classes and methods are documented
- **CSS Custom Properties** — full theming via CSS variables, instant theme switch
- **`@keyframes` animations** — smooth slide-in / slide-out for task items
- **`prefers-color-scheme`** — auto-applies dark mode based on OS preference
- **Keyboard accessible** — all interactive elements reachable and operable via keyboard
