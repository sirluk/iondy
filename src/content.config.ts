import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const posts = defineCollection({
  loader: glob({
    base: './content',
    pattern: '*/index.md',
    generateId: ({ entry }) => entry.replace(/\/index\.md$/, ''),
  }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string(),
    category: z.enum(['Research', 'Engineering', 'Learning']).default('Learning'),
    tags: z.array(z.string()).default([]),
    original_url: z.string().url().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { posts };
