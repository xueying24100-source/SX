/**
 * @file drag.js
 * @description HTML5 Drag-and-Drop reordering logic for the To-Do List application.
 */

/**
 * Manages drag-and-drop reordering of task list items.
 * Uses the native HTML5 Drag and Drop API for maximum compatibility.
 */
class DragManager {
  /**
   * @param {HTMLElement} listEl - The <ul> element that contains task items
   * @param {function(string[]): void} onReorder - Callback invoked with the new ordered array of task IDs
   */
  constructor(listEl, onReorder) {
    /** @type {HTMLElement} */
    this.listEl = listEl;
    /** @type {function(string[]): void} */
    this.onReorder = onReorder;
    /** @type {HTMLElement|null} */
    this._dragging = null;
    /** @type {HTMLElement|null} */
    this._placeholder = null;

    this._bindEvents();
  }

  /** Attaches drag event listeners to the list via event delegation. */
  _bindEvents() {
    this.listEl.addEventListener('dragstart', this._onDragStart.bind(this));
    this.listEl.addEventListener('dragover', this._onDragOver.bind(this));
    this.listEl.addEventListener('dragenter', this._onDragEnter.bind(this));
    this.listEl.addEventListener('dragleave', this._onDragLeave.bind(this));
    this.listEl.addEventListener('drop', this._onDrop.bind(this));
    this.listEl.addEventListener('dragend', this._onDragEnd.bind(this));
  }

  /**
   * Returns the closest .task-item ancestor (or self) of the given element.
   * @param {HTMLElement} el
   * @returns {HTMLElement|null}
   */
  _getTaskItem(el) {
    return el.closest('.task-item');
  }

  /** @param {DragEvent} e */
  _onDragStart(e) {
    const item = this._getTaskItem(e.target);
    if (!item) return;

    this._dragging = item;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', item.dataset.id);

    // Create placeholder with same height
    this._placeholder = document.createElement('li');
    this._placeholder.className = 'task-placeholder';
    this._placeholder.style.height = `${item.offsetHeight}px`;

    // Slight delay so the drag image is captured before opacity change
    requestAnimationFrame(() => {
      item.classList.add('dragging');
    });
  }

  /** @param {DragEvent} e */
  _onDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';

    const target = this._getTaskItem(e.target);
    if (!target || target === this._dragging || target === this._placeholder) return;

    const rect = target.getBoundingClientRect();
    const midY = rect.top + rect.height / 2;

    if (e.clientY < midY) {
      this.listEl.insertBefore(this._placeholder, target);
    } else {
      this.listEl.insertBefore(this._placeholder, target.nextSibling);
    }
  }

  /** @param {DragEvent} e */
  _onDragEnter(e) {
    e.preventDefault();
  }

  /** @param {DragEvent} e */
  _onDragLeave(e) {
    // Only remove placeholder if the cursor leaves the list entirely
    if (!this.listEl.contains(e.relatedTarget)) {
      this._placeholder && this._placeholder.remove();
    }
  }

  /** @param {DragEvent} e */
  _onDrop(e) {
    e.preventDefault();
    if (!this._dragging || !this._placeholder) return;

    this.listEl.insertBefore(this._dragging, this._placeholder);
    this._cleanup();

    // Collect new order of IDs from visible DOM items
    const orderedIds = Array.from(
      this.listEl.querySelectorAll('.task-item[data-id]')
    ).map(el => el.dataset.id);

    this.onReorder(orderedIds);
  }

  /** @param {DragEvent} e */
  _onDragEnd(e) {
    this._cleanup();
  }

  /** Removes drag-state classes and placeholder element. */
  _cleanup() {
    if (this._dragging) {
      this._dragging.classList.remove('dragging');
      this._dragging = null;
    }
    if (this._placeholder) {
      this._placeholder.remove();
      this._placeholder = null;
    }
    // Remove any lingering drag-over highlights
    this.listEl.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
  }
}
