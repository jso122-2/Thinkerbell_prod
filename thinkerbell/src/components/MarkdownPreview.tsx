interface MarkdownPreviewProps {
  content: string;
  className?: string;
}

export function MarkdownPreview({ content, className = '' }: MarkdownPreviewProps) {
  // Simple markdown-to-HTML converter for basic formatting
  const convertMarkdown = (markdown: string) => {
    return markdown
      // Headers
      .replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold text-black mt-6 mb-3">$1</h3>')
      .replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold text-black mt-8 mb-4">$1</h2>')
      .replace(/^# (.*$)/gim, '<h1 class="text-2xl font-black text-black mt-8 mb-6">$1</h1>')
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold">$1</strong>')
      // Italic
      .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
      // Code blocks
      .replace(/```[\s\S]*?```/g, '<pre class="bg-gray-100 border border-gray-200 rounded-lg p-4 my-4 overflow-x-auto"><code class="text-sm font-mono">$&</code></pre>')
      // Inline code
      .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-2 py-1 rounded text-sm font-mono">$1</code>')
      // Horizontal rules
      .replace(/^---$/gim, '<hr class="my-6 border-t border-gray-300" />')
      // Line breaks
      .replace(/\n/g, '<br />');
  };

  return (
    <div className={`prose max-w-none p-6 ${className}`}>
      <div 
        className="markdown-content"
        dangerouslySetInnerHTML={{ 
          __html: convertMarkdown(content) 
        }}
      />
    </div>
  );
}