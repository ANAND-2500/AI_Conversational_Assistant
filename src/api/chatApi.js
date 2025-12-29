export async function sendMessage(text) {
  await new Promise(r => setTimeout(r, 1200));
  return "Reply to: " + text;
}
