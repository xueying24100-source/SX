/**
 * @file app.js
 * @description Main controller for the To-Do List application.
 * Connects Model (tasks array) ↔ View (UIManager) ↔ Storage (StorageManager).
 */

/**
 * Main application controller class.
 * Follows an MVC-like pattern: this class acts as the Controller,
 * UIManager is the View, and StorageManager is the Model persistence layer.
 */
class TodoApp {
  constructor() {
    /** @type {StorageManager} */
    this.storage = new StorageManager();

    /** @type {Array<{id: string, text: string, completed: boolean, createdAt: string, dueDate: string|null}>} */
    this.tasks = [];

    /** @type {'all'|'active'|'completed'} */
    this.filter = 'all';

    /** @type {UIManager} */
    this.ui = new UIManager({
      onDelete: id => this._deleteTask(id),
      onToggle: id => this._toggleTask(id),
      onEdit: (id, text) => this._editTask(id, text),
      onFilterChange: filter => this._setFilter(filter),
      onClearCompleted: () => this._clearCompleted(),
      onThemeToggle: () => this._toggleTheme(),
    });

    /** @type {DragManager} */
    this.drag = new DragManager(
      document.getElementById('task-list'),
      orderedIds => this._reorderTasks(orderedIds)
    );

    this._init();
  }

  /** Initializes the application: loads tasks and theme from storage and renders. */
  _init() {
    // Load tasks
    this.tasks = this.storage.getTasks();

    // Apply persisted or system theme
    const savedTheme = this.storage.getTheme();
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');
    this.ui.toggleTheme(theme);

    // Bind the add-task form
    const form = document.getElementById('task-form');
    form.addEventListener('submit', e => {
      e.preventDefault();
      this._addTask();
    });

    this._render();
  }

  /**
   * Reads the form inputs and adds a new task to the model.
   * Validates that text is not empty.
   */
  _addTask() {
    const input = document.getElementById('task-input');
    const dueDateInput = document.getElementById('due-date-input');

    const text = input.value.trim();
    if (!text) {
      this.ui.showError('Task cannot be empty.');
      input.focus();
      return;
    }

    /** @type {{id: string, text: string, completed: boolean, createdAt: string, dueDate: string|null}} */
    const newTask = {
      id: `task-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      text,
      completed: false,
      createdAt: new Date().toISOString(),
      dueDate: dueDateInput.value || null,
    };

    this.tasks.push(newTask);
    const saved = this.storage.saveTasks(this.tasks);
    if (!saved) {
      this.ui.showError('Storage quota exceeded. Could not save task.');
      this.tasks.pop();
      return;
    }

    input.value = '';
    dueDateInput.value = '';
    this._render();
    input.focus();
  }

  /**
   * Removes a task by its id.
   * @param {string} id
   */
  _deleteTask(id) {
    this.tasks = this.tasks.filter(t => t.id !== id);
    this.storage.saveTasks(this.tasks);
    this._render();
  }

  /**
   * Toggles the completed state of a task.
   * @param {string} id
   */
  _toggleTask(id) {
    this.tasks = this.tasks.map(t =>
      t.id === id ? { ...t, completed: !t.completed } : t
    );
    this.storage.saveTasks(this.tasks);
    this._render();
  }

  /**
   * Updates the text of an existing task.
   * @param {string} id
   * @param {string} newText
   */
  _editTask(id, newText) {
    this.tasks = this.tasks.map(t =>
      t.id === id ? { ...t, text: newText } : t
    );
    this.storage.saveTasks(this.tasks);
    this._render();
  }

  /**
   * Changes the active filter and re-renders.
   * @param {'all'|'active'|'completed'} filter
   */
  _setFilter(filter) {
    this.filter = filter;
    this._render();
  }

  /** Removes all completed tasks from the model. */
  _clearCompleted() {
    this.tasks = this.tasks.filter(t => !t.completed);
    this.storage.saveTasks(this.tasks);
    this._render();
  }

  /**
   * Re-orders the tasks array based on the new DOM order of IDs.
   * Called by DragManager after a successful drop.
   * @param {string[]} orderedIds - Array of task IDs in their new order
   */
  _reorderTasks(orderedIds) {
    // Build a lookup map for O(1) access
    const taskMap = new Map(this.tasks.map(t => [t.id, t]));

    // Rebuild ordered array; append any tasks not currently visible (filtered out)
    const visible = orderedIds.map(id => taskMap.get(id)).filter(Boolean);
    const hiddenIds = new Set(this.tasks.map(t => t.id));
    orderedIds.forEach(id => hiddenIds.delete(id));
    const hidden = Array.from(hiddenIds).map(id => taskMap.get(id)).filter(Boolean);

    this.tasks = [...visible, ...hidden];
    this.storage.saveTasks(this.tasks);
  }

  /** Toggles between light and dark themes, persisting the choice. */
  _toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    this.ui.toggleTheme(next);
    this.storage.saveTheme(next);
  }

  /** Re-renders the task list and counter based on current state. */
  _render() {
    this.ui.renderTasks(this.tasks, this.filter);
    const activeCount = this.tasks.filter(t => !t.completed).length;
    this.ui.renderCounter(activeCount);
  }
}

// Bootstrap the application after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  window.todoApp = new TodoApp();
});
