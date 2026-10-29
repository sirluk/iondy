/** Add designs here, then import their CSS in Base.astro. Both preview UIs use this list. */
export const directions = [
  {
    id: 'editorial',
    name: 'Editorial',
    description: 'Warm white, serif headings, and a spacious layout with writing and projects alongside each other.',
  },
  {
    id: 'index',
    name: 'Index',
    description: 'Crisp white, restrained blue accents, a navigation rail, and projects before the writing.',
  },
  {
    id: 'paper',
    name: 'Paper',
    description: 'White, monochrome, and a single reading column. The simplest setting for text, code, and figures.',
  },
  {
    id: 'margin',
    name: 'Margin',
    description: 'A research-paper grid in IBM Plex. Labels, dates, filters, and contents sit in a margin column; figures and code use the full width.',
  },
  {
    id: 'quiet',
    name: 'Quiet',
    description: 'One narrow column in Geist, small type, and whitespace instead of rules. Navigation sits beside the name; lists are title and date.',
  },
  {
    id: 'folio',
    name: 'Folio',
    description: 'Centered and book-like, set entirely in Source Serif with small-caps navigation, generous margins, and wide figures.',
  },
];

/** The design visitors see first. */
export const defaultLook = 'quiet';

/** Names for the AI slop slider's stops in Quiet. Level 0 is plain Quiet; src/styles/slop.css styles levels 1 and up. */
export const slopLevels = ['Off', 'Mild', 'Medium', 'Spicy', 'Hot', 'Maximum'];
