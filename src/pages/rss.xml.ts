import { getPosts } from '../lib/content';
const escape = (value: string) => value.replace(/[<>&"']/g, (char) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' }[char]!));
export async function GET() {
  const posts = await getPosts();
  const items = posts.map((post) => `<item><title>${escape(post.data.title)}</title><link>https://iondy.com/writing/${post.id}/</link><guid>https://iondy.com/writing/${post.id}/</guid><pubDate>${post.data.date.toUTCString()}</pubDate><description>${escape(post.data.description)}</description></item>`).join('');
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>iondy — Lukas Hauzenberger</title><link>https://iondy.com</link><description>Projects, research, and things learned along the way.</description><language>en</language>${items}</channel></rss>`, { headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' } });
}
