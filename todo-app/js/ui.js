/**
 * @file ui.js
 * @description DOM manipulation and rendering layer for the To-Do List application.
 */

/**
 * Manages all UI rendering and DOM interactions.
 * Responsible for creating, updating, and removing DOM elements.
 */
class UIManager {
  /**
   * @param {Object} callbacks
   * @param {function(string): void} callbacks.onDelete - Called with task id when delete is clicked
   * @param {function(string): void} callbacks.onToggle - Called with task id when checkbox changes
   * @param {function(string, string): void} callbacks.onEdit - Called with (id, newText) after inline edit
   * @param {function(string): void} callbacks.onFilterChange - Called with filter name
   * @param {function(): void} callbacks.onClearCompleted - Called when clear-completed is clicked
   * @param {function(): void} callbacks.onThemeToggle - Called when theme toggle is clicked
   */
  constructor(callbacks) {
    this.callbacks = callbacks;

    /** @type {HTMLElement} */
    this.taskList = document.getElementById('task-list');
    /** @type {HTMLElement} */
    this.counterEl = document.getElementById('task-counter');
    /** @type {HTMLElement} */
    this.errorEl = document.getElementById('error-message');
    /** @type {HTMLElement} */
    this.filterButtons = document.querySelectorAll('.filter-btn');
    /** @type {HTMLButtonElement} */
    this.clearCompletedBtn = document.getElementById('clear-completed');
    /** @type {HTMLButtonElement} */
    this.themeToggleBtn = document.getElementById('theme-toggle');

    this._bindStaticEvents();
  }

  /** Attaches event listeners that don't change with re-renders. */
  _bindStaticEvents() {
    this.filterButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        this.filterButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.callbacks.onFilterChange(btn.dataset.filter);
      });
    });

    this.clearCompletedBtn.addEventListener('click', () => {
      this.callbacks.onClearCompleted();
    });

    this.themeToggleBtn.addEventListener('click', () => {
      this.callbacks.onThemeToggle();
    });
  }

  /**
   * Renders the full task list into the DOM.
   * @param {Array<{id: string, text: string, completed: boolean, createdAt: string, dueDate: string|null}>} tasks
   * @param {'all'|'active'|'completed'} filter
   */
  renderTasks(tasks, filter) {
    const filtered = tasks.filter(task => {
      if (filter === 'active') return !task.completed;
      if (filter === 'completed') return task.completed;
      return true;
    });

    this.taskList.innerHTML = '';

    if (filtered.length === 0) {
      const empty = document.createElement('li');
      empty.className = 'task-empty';
      empty.textContent = filter === 'completed'
        ? 'No completed tasks yet.'
        : filter === 'active'
          ? 'All tasks are done! 🎉'
          : 'No tasks yet. Add one above!';
      this.taskList.appendChild(empty);
      return;
    }

    filtered.forEach(task => {
      const li = this._createTaskElement(task);
      this.taskList.appendChild(li);
    });
  }

  /**
   * Creates a single task <li> element with all controls.
   * @param {{id: string, text: string, completed: boolean, createdAt: string, dueDate: string|null}} task
   * @returns {HTMLLIElement}
   */
  _createTaskElement(task) {
    const li = document.createElement('li');
    li.className = `task-item${task.completed ? ' completed' : ''}`;
    li.dataset.id = task.id;
    li.setAttribute('draggable', 'true');
    li.setAttribute('aria-label', `Task: ${task.text}`);

    const isOverdue = task.dueDate && !task.completed && new Date(task.dueDate) < new Date();

    li.innerHTML = `
      <span class="drag-handle" aria-hidden="true" title="Drag to reorder">⠿</span>

      <input
        type="checkbox"
        class="task-checkbox"
        id="chk-${task.id}"
        aria-label="Mark '${task.text}' as ${task.completed ? 'incomplete' : 'complete'}"
        ${task.completed ? 'checked' : ''}
      />

      <div class="task-content">
        <label class="task-text" for="chk-${task.id}">${this._escapeHtml(task.text)}</label>
        <input
          type="text"
          class="task-edit-input"
          value="${this._escapeAttr(task.text)}"
          aria-label="Edit task text"
          style="display:none"
        />
        ${task.dueDate ? `<span class="task-due${isOverdue ? ' overdue' : ''}" title="Due date">${isOverdue ? '⚠ ' : '📅 '}${this._formatDate(task.dueDate)}</span>` : ''}
      </div>

      <div class="task-actions">
        <button class="btn-edit" data-id="${task.id}" aria-label="Edit task" title="Edit">✏️</button>
        <button class="btn-delete" data-id="${task.id}" aria-label="Delete task" title="Delete">🗑️</button>
      </div>
    `;

    this._bindTaskEvents(li, task);
    return li;
  }

  /**
   * Binds all interaction events to a task element.
   * @param {HTMLLIElement} li
   * @param {{id: string, text: string, completed: boolean}} task
   */
  _bindTaskEvents(li, task) {
    const checkbox = li.querySelector('.task-checkbox');
    const editBtn = li.querySelector('.btn-edit');
    const deleteBtn = li.querySelector('.btn-delete');
    const textLabel = li.querySelector('.task-text');
    const editInput = li.querySelector('.task-edit-input');

    checkbox.addEventListener('change', () => this.callbacks.onToggle(task.id));
    deleteBtn.addEventListener('click', () => {
      li.classList.add('removing');
      li.addEventListener('animationend', () => this.callbacks.onDelete(task.id), { once: true });
    });

    // Toggle edit mode
    const startEdit = () => {
      textLabel.style.display = 'none';
      editInput.style.display = 'block';
      editInput.focus();
      editInput.select();
      editBtn.setAttribute('aria-label', 'Save task');
      editBtn.title = 'Save';
      editBtn.textContent = '💾';
    };

    const commitEdit = () => {
      const newText = editInput.value.trim();
      if (newText && newText !== task.text) {
        this.callbacks.onEdit(task.id, newText);
      } else {
        // Revert UI without saving
        editInput.style.display = 'none';
        textLabel.style.display = '';
        editBtn.textContent = '✏️';
        editBtn.title = 'Edit';
      }
    };

    editBtn.addEventListener('click', () => {
      if (editInput.style.display === 'none') {
        startEdit();
      } else {
        commitEdit();
      }
    });

    textLabel.addEventListener('dblclick', startEdit);

    editInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') commitEdit();
      if (e.key === 'Escape') {
        editInput.style.display = 'none';
        textLabel.style.display = '';
        editBtn.textContent = '✏️';
        editBtn.title = 'Edit';
      }
    });

    /** Delay (ms) to allow the save-button click event to fire before blur resets edit mode. */
    const EDIT_BLUR_DELAY = 150;
    editInput.addEventListener('blur', () => {
      setTimeout(() => {
        if (editInput.style.display !== 'none') {
          commitEdit();
        }
      }, EDIT_BLUR_DELAY);
    });
  }

  /**
   * Updates the remaining active task counter display.
   * @param {number} count
   */
  renderCounter(count) {
    this.counterEl.textContent = `${count} item${count !== 1 ? 's' : ''} left`;
  }

  /**
   * Shows an error message to the user, auto-hides after 3 seconds.
   * @param {string} msg
   */
  showError(msg) {
    this.errorEl.textContent = msg;
    this.errorEl.removeAttribute('hidden');
    this.errorEl.classList.add('visible');
    clearTimeout(this._errorTimer);
    this._errorTimer = setTimeout(() => {
      this.errorEl.classList.remove('visible');
      this.errorEl.setAttribute('hidden', '');
    }, 3000);
  }

  /**
   * Updates the theme toggle button icon and the html data-theme attribute.
   * @param {'light'|'dark'} theme
   */
  toggleTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    this.themeToggleBtn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    this.themeToggleBtn.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    this.themeToggleBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
  }

  /**
   * Escapes HTML special characters to prevent XSS.
   * @param {string} str
   * @returns {string}
   */
  _escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  /**
   * Escapes a string for use inside an HTML attribute value.
   * @param {string} str
   * @returns {string}
   */
  _escapeAttr(str) {
    return str.replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }

  /**
   * Formats an ISO date string to a human-readable short date.
   * @param {string} isoDate
   * @returns {string}
   */
  _formatDate(isoDate) {
    const d = new Date(isoDate);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  }
}
