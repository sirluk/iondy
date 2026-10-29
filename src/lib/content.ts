import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

export async function getPosts(): Promise<Post[]> {
  return (await getCollection('posts', ({ data }) => !data.draft))
    .sort((a, b) => b.data.date.getTime() - a.data.date.getTime());
}

export function formatDate(date: Date, short = false) {
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric', month: short ? 'short' : 'long', year: 'numeric', timeZone: 'UTC',
  }).format(date);
}

export function readingTime(post: Post) {
  return Math.max(1, Math.ceil((post.body || '').split(/\s+/).length / 220));
}

export const projects = [
  {
    name: 'pytora',
    kind: 'Parameter-efficient learning',
    description: 'LoRA, with just PyTorch. A small implementation built for flexibility and experimentation.',
    url: 'https://github.com/sirluk/pytora',
    note: 'introducing-pytora-a-minimal-pytorch-implementation-of-lora',
    stack: 'Python · PyTorch',
    mark: 'layers',
  },
  {
    name: 'llm_finetuning',
    kind: 'Experiments & implementations',
    description: 'Practical experiments with language models, from efficient attention masks to multilabel classification.',
    url: 'https://github.com/sirluk/llm_finetuning',
    note: 'efficient-llm-pretraining-packed-sequences-and-masked-attention',
    stack: 'Python · Language models',
    mark: 'grid',
  },
  {
    name: 'vertex_ai_pytorch',
    kind: 'From training to deployment',
    description: 'Taking PyTorch models into Google Cloud with automated training and deployment pipelines.',
    url: 'https://github.com/sirluk/vertex_ai_pytorch',
    note: 'training-and-serving-pytorch-models-in-the-google-cloud-with-vertex-ai-pipelines',
    stack: 'PyTorch · Google Cloud',
    mark: 'pipeline',
  },
];
