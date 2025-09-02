import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search, Copy, Filter, RotateCcw, Play, Pause, ArrowDown, ArrowUp } from 'lucide-react';
import { getRunLogs } from '../lib/api';
import type { LogLine } from '../lib/schemas';

interface LogViewerProps {
  runId: string;
  className?: string;
}

export function LogViewer({ runId, className = '' }: LogViewerProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [levelFilter, setLevelFilter] = useState<LogLine['level'] | 'ALL'>('ALL');
  const [isTailing, setIsTailing] = useState(true);
  const [offset, setOffset] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const { data: logResponse, isLoading, error } = useQuery({
    queryKey: ['logs', runId, offset, levelFilter, searchQuery],
    queryFn: () => getRunLogs(runId, {
      offset,
      limit: 100,
      level: levelFilter === 'ALL' ? undefined : levelFilter,
      q: searchQuery || undefined
    }),
    refetchInterval: isTailing ? 2000 : false, // Auto-refresh when tailing
    keepPreviousData: true
  });

  const logs = logResponse?.lines || [];
  const total = logResponse?.total || 0;

  // Auto-scroll to bottom when tailing
  useEffect(() => {
    if (isTailing && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isTailing]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      
      switch (e.key.toLowerCase()) {
        case 'f':
          e.preventDefault();
          document.getElementById('log-search')?.focus();
          break;
        case 'escape':
          setSearchQuery('');
          break;
        case 'g':
          if (e.shiftKey) {
            // Shift+G - go to bottom
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
          } else {
            // g - go to top
            scrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  const highlightText = useCallback((text: string, query: string) => {
    if (!query) return text;
    
    const regex = new RegExp(`(${query})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <span key={index} className="bg-yellow-200 text-yellow-900 font-bold">
          {part}
        </span>
      ) : part
    );
  }, []);

  const copyLogs = useCallback(() => {
    const logText = logs.map(log => 
      `${log.timestamp} [${log.level}] ${log.message}`
    ).join('\n');
    
    navigator.clipboard.writeText(logText).then(() => {
      // Could add a toast notification here
      console.log('Logs copied to clipboard');
    });
  }, [logs]);

  const getLevelColor = (level: LogLine['level']) => {
    switch (level) {
      case 'ERROR': return 'text-red-600 bg-red-50 border-red-200';
      case 'WARN': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'INFO': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'DEBUG': return 'text-gray-600 bg-gray-50 border-gray-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  if (error) {
    return (
      <div className={`p-6 ${className}`}>
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <h3 className="text-sm font-medium text-red-800">Failed to load logs</h3>
          <p className="text-sm text-red-700 mt-1">
            {error instanceof Error ? error.message : 'An error occurred'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header Controls */}
      <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-black text-black">Logs</h3>
          <span className="text-sm text-gray-500">
            {logs.length} of {total} lines
          </span>
        </div>
        
        <div className="flex items-center space-x-3">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              id="log-search"
              type="text"
              placeholder="Search logs... (press 'f')"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent text-sm"
            />
          </div>

          {/* Level Filter */}
          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 text-sm"
          >
            <option value="ALL">All Levels</option>
            <option value="ERROR">❌ Error</option>
            <option value="WARN">⚠️ Warning</option>
            <option value="INFO">ℹ️ Info</option>
            <option value="DEBUG">🔍 Debug</option>
          </select>

          {/* Controls */}
          <button
            onClick={() => setIsTailing(!isTailing)}
            className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              isTailing 
                ? 'bg-green-100 text-green-700 border border-green-200' 
                : 'bg-gray-100 text-gray-700 border border-gray-200'
            }`}
          >
            {isTailing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isTailing ? 'Pause' : 'Resume'}</span>
          </button>

          <button
            onClick={copyLogs}
            className="flex items-center space-x-2 px-3 py-2 bg-gray-100 text-gray-700 border border-gray-200 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
          >
            <Copy className="w-4 h-4" />
            <span>Copy</span>
          </button>
        </div>
      </div>

      {/* Keyboard Shortcuts Help */}
      <div className="px-4 py-2 bg-blue-50 border-b border-blue-200 text-xs text-blue-700">
        <strong>Shortcuts:</strong> F = search, Esc = clear search, G = top, Shift+G = bottom
      </div>

      {/* Log Content */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-auto bg-black text-green-400 font-mono text-sm"
        style={{ fontFamily: 'SF Mono, Monaco, Consolas, monospace' }}
      >
        {isLoading && logs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex items-center space-x-3 text-green-400">
              <RotateCcw className="w-5 h-5 animate-spin" />
              <span>Loading logs...</span>
            </div>
          </div>
        ) : logs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-2xl mb-2">📝</div>
              <p>No logs found</p>
              {searchQuery && <p className="text-sm mt-1">Try adjusting your search or filter</p>}
            </div>
          </div>
        ) : (
          <div className="p-4 space-y-1">
            {logs.map((log, index) => (
              <div key={index} className="flex items-start space-x-4 hover:bg-gray-900 px-2 py-1 rounded">
                <span className="text-gray-500 text-xs font-medium w-24 flex-shrink-0">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className={`text-xs font-bold w-12 flex-shrink-0 ${
                  log.level === 'ERROR' ? 'text-red-400' :
                  log.level === 'WARN' ? 'text-yellow-400' :
                  log.level === 'INFO' ? 'text-blue-400' :
                  'text-gray-400'
                }`}>
                  {log.level}
                </span>
                <span className="flex-1 break-words">
                  {highlightText(log.message, searchQuery)}
                </span>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Navigation Controls */}
      {total > logs.length && (
        <div className="flex items-center justify-between p-4 bg-gray-50 border-t border-gray-200">
          <span className="text-sm text-gray-600">
            Showing {logs.length} of {total} logs
          </span>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setOffset(Math.max(0, offset - 100))}
              disabled={offset === 0}
              className="flex items-center space-x-1 px-3 py-1 bg-white border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ArrowUp className="w-4 h-4" />
              <span>Newer</span>
            </button>
            <button
              onClick={() => setOffset(offset + 100)}
              disabled={!logResponse?.nextOffset}
              className="flex items-center space-x-1 px-3 py-1 bg-white border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span>Older</span>
              <ArrowDown className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}