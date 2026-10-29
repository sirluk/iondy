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

/** Shown on src/pages/_projects.astro, which is offline for now. */
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

export const scholarUrl = 'https://scholar.google.com/citations?user=fpj_6tYAAAAJ';

export type Paper = {
  title: string;
  venue: string;
  year: number;
  /** Omitted when the author list isn't public yet. */
  authors?: string[];
  note?: string;
  summary: string;
  /** The first link is the paper itself; the title links there. */
  links: { label: string; url: string }[];
};

/** Publications, in the order of the CV. */
export const papers: Paper[] = [
  {
    title: 'Effective Distillation to Hybrid xLSTM Architectures',
    venue: 'ICML',
    year: 2026,
    authors: ['Lukas Hauzenberger', 'Niklas Schmidinger', 'Thomas Schmied', 'Anamaria-Roberta Hartl', 'David Stap', 'Pieter-Jan Hoedt', 'Maximilian Beck', 'Sebastian Böck', 'Günter Klambauer', 'Sepp Hochreiter'],
    summary: 'Near-lossless linearization of 7–8B models into hybrid xLSTM architectures, with up to 40k context and 2D parallelism (FSDP + TP).',
    links: [{ label: 'arXiv', url: 'https://arxiv.org/abs/2603.15590' }, { label: 'OpenReview', url: 'https://openreview.net/forum?id=kj9wufh0EI' }],
  },
  {
    title: 'KVpop — Key-Value Cache Compression with Predictive Online Pruning',
    venue: 'ICLR TTU Workshop',
    year: 2026,
    authors: ['Lukas Hauzenberger', 'Niklas Schmidinger', 'Anamaria-Roberta Hartl', 'David Stap', 'Thomas Schmied', 'Sebastian Böck', 'Günter Klambauer', 'Sepp Hochreiter'],
    summary: 'Efficient long-context reasoning that retains most of the accuracy of dense attention on math benchmarks.',
    links: [{ label: 'arXiv', url: 'https://arxiv.org/abs/2607.05061' }, { label: 'Model', url: 'https://huggingface.co/sirluk/Qwen3-8B-KVpop-4x' }],
  },
  {
    title: 'Parameter-Efficient Fine-Tuning via Explained Variance Adaptation',
    venue: 'NeurIPS',
    year: 2025,
    authors: ['Fabian Paischer', 'Lukas Hauzenberger', 'Thomas Schmied', 'Benedikt Alkin', 'Marc Peter Deisenroth', 'Sepp Hochreiter'],
    note: 'Shared first authorship',
    summary: 'Data-driven LoRA initialization, with a fast GPU implementation of incremental PCA.',
    links: [{ label: 'arXiv', url: 'https://arxiv.org/abs/2410.07170' }, { label: 'OpenReview', url: 'https://openreview.net/forum?id=movsqor65f' }, { label: 'Code', url: 'https://github.com/ml-jku/EVA' }],
  },
  {
    title: 'Modular and On-demand Bias Mitigation with Attribute-Removal Subnetworks',
    venue: 'Findings of ACL',
    year: 2023,
    authors: ['Lukas Hauzenberger', 'Shahed Masoudian', 'Deepak Kumar', 'Markus Schedl', 'Navid Rekabsaz'],
    summary: 'Unlearning biased representations in pre-trained language models with attribute-removal subnetworks.',
    links: [{ label: 'arXiv', url: 'https://arxiv.org/abs/2205.15171' }, { label: 'ACL Anthology', url: 'https://aclanthology.org/2023.findings-acl.386/' }],
  },
  {
    title: 'Capability Retention under Post-Training Linearization of Small LMs',
    venue: 'NeurIPS SLM-Agents Workshop',
    year: 2026,
    summary: 'Distilling 0.6–4B Qwen3 and SmolLM3 models into hybrid architectures, with 3.2–7.1× throughput on long generations.',
    links: [{ label: 'OpenReview', url: 'https://openreview.net/forum?id=sM5teckKt2' }],
  },
  {
    title: 'On Subquadratic Architectures: From Applications to Principles',
    venue: 'arXiv preprint',
    year: 2026,
    authors: ['Anamaria-Roberta Hartl', 'Levente Zólyomi', 'David Stap', 'Pieter-Jan Hoedt', 'Niklas Schmidinger', 'Lukas Hauzenberger', 'Sebastian Böck', 'Günter Klambauer', 'Sepp Hochreiter'],
    summary: 'A comparison of xLSTM, Mamba-2, and Gated DeltaNet for pre-training and distillation.',
    links: [{ label: 'arXiv', url: 'https://arxiv.org/abs/2606.12364' }],
  },
];
