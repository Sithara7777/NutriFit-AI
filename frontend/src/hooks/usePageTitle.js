import { useEffect } from 'react';

/**
 * Set the document title for the current view.
 *
 * A single-page application keeps one `<title>` for its whole lifetime unless
 * something updates it. Screen-reader users rely on the title to know where
 * they have navigated to, and it is what browser history and tab lists show
 * (WCAG 2.4.2 Page Titled).
 */
export function usePageTitle(title) {
  useEffect(() => {
    const previous = document.title;
    document.title = title ? `${title} · NutriFit-AI` : 'NutriFit-AI';
    return () => {
      document.title = previous;
    };
  }, [title]);
}
