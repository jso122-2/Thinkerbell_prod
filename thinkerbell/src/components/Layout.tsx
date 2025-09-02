import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Brain, BookOpen, FileText, Zap } from 'lucide-react';
import { clsx } from 'clsx';

interface LayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: 'Home', href: '/', icon: Home, shortcut: 'g h', description: 'Dashboard & Overview' },
  { name: 'Model', href: '/model', icon: Brain, shortcut: 'g m', description: 'Train & Deploy Models' },
  { name: 'Template', href: '/template', icon: FileText, shortcut: 'g t', description: 'Generate Agreements' },
  { name: 'Examples', href: '/examples', icon: BookOpen, shortcut: 'g e', description: 'Browse Past Success' },
];

export function Layout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-white flex">
      {/* Sidebar */}
      <div className="flex flex-col w-80 bg-black shadow-2xl">
        <div className="flex items-center flex-shrink-0 px-8 py-8 border-b border-gray-800">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-tb-magenta rounded-full flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-3xl font-black text-white">
              Thinker<span className="tb-accent-pink">bell</span>
            </h1>
          </div>
        </div>
        
        {/* Zigzag accent */}
        <div className="px-8 py-4">
          <div className="tb-zigzag"></div>
        </div>
        
        <nav className="flex-1 px-6 py-4 space-y-3">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={clsx(
                  'group flex flex-col px-6 py-5 text-lg font-bold rounded-xl transition-all duration-200',
                  isActive
                    ? 'bg-tb-magenta text-white shadow-lg transform scale-105'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white hover:transform hover:scale-105'
                )}
              >
                <div className="flex items-center w-full">
                  <item.icon
                    className={clsx(
                      'mr-4 flex-shrink-0 h-6 w-6',
                      isActive ? 'text-white' : 'text-gray-400 group-hover:text-white'
                    )}
                  />
                  <span className="flex-1">{item.name}</span>
                  {item.shortcut && (
                    <span className="text-xs opacity-75 bg-gray-700 px-2 py-1 rounded">
                      {item.shortcut}
                    </span>
                  )}
                </div>
                <p className={clsx(
                  'text-sm font-medium mt-1 ml-10 transition-opacity',
                  isActive ? 'text-pink-100' : 'text-gray-400'
                )}>
                  {item.description}
                </p>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-6 border-t border-gray-800">
          <div className="text-xs text-gray-500 space-y-1">
            <p className="font-bold text-white">We are Thinkers.</p>
            <p className="font-bold text-white">And we are Tinkers.</p>
            <div className="tb-zigzag-green mt-3"></div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-auto bg-white">
        {children}
      </div>
    </div>
  );
}
