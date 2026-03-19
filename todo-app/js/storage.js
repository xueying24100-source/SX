/**
 * @file storage.js
 * @description localStorage abstraction layer for the To-Do List application.
 */

/**
 * Manages all interactions with the browser's localStorage API.
 * Handles JSON serialization/deserialization and storage quota errors.
 */
class StorageManager {
  /** @type {string} */
  static TASKS_KEY = 'todoApp_tasks';

  /** @type {string} */
  static THEME_KEY = 'todoApp_theme';

  /**
   * Retrieves the tasks array from localStorage.
   * @returns {Array<{id: string, text: string, completed: boolean, createdAt: string, dueDate: string|null}>}
   */
  getTasks() {
    try {
      const raw = localStorage.getItem(StorageManager.TASKS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (err) {
      console.error('[StorageManager] Failed to parse tasks from localStorage:', err);
      return [];
    }
  }

  /**
   * Persists the tasks array to localStorage.
   * @param {Array<{id: string, text: string, completed: boolean, createdAt: string, dueDate: string|null}>} tasks
   * @returns {boolean} true on success, false on failure (e.g. quota exceeded)
   */
  saveTasks(tasks) {
    try {
      localStorage.setItem(StorageManager.TASKS_KEY, JSON.stringify(tasks));
      return true;
    } catch (err) {
      if (err instanceof DOMException && (
        err.code === 22 /* QUOTA_EXCEEDED_ERR (legacy DOM error code) */ ||
        err.code === 1014 /* NS_ERROR_DOM_QUOTA_REACHED (Firefox legacy code) */ ||
        err.name === 'QuotaExceededError' ||
        err.name === 'NS_ERROR_DOM_QUOTA_REACHED'
      )) {
        console.error('[StorageManager] Storage quota exceeded:', err);
        return false;
      }
      console.error('[StorageManager] Failed to save tasks:', err);
      return false;
    }
  }

  /**
   * Retrieves the saved theme preference from localStorage.
   * @returns {'light'|'dark'|null}
   */
  getTheme() {
    try {
      return localStorage.getItem(StorageManager.THEME_KEY);
    } catch (err) {
      console.error('[StorageManager] Failed to retrieve theme:', err);
      return null;
    }
  }

  /**
   * Saves the theme preference to localStorage.
   * @param {'light'|'dark'} theme
   */
  saveTheme(theme) {
    try {
      localStorage.setItem(StorageManager.THEME_KEY, theme);
    } catch (err) {
      console.error('[StorageManager] Failed to save theme:', err);
    }
  }
}
