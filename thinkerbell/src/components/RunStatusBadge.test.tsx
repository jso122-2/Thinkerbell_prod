import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RunStatusBadge } from './RunStatusBadge';

describe('RunStatusBadge', () => {
  it('renders success status correctly', () => {
    render(<RunStatusBadge status="success" />);
    
    const badge = screen.getByText('success');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-success-50', 'text-success-700');
  });

  it('renders running status correctly', () => {
    render(<RunStatusBadge status="running" />);
    
    const badge = screen.getByText('running');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-warning-50', 'text-warning-700');
  });

  it('renders failed status correctly', () => {
    render(<RunStatusBadge status="failed" />);
    
    const badge = screen.getByText('failed');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-error-50', 'text-error-700');
  });

  it('renders queued status correctly', () => {
    render(<RunStatusBadge status="queued" />);
    
    const badge = screen.getByText('queued');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-gray-100', 'text-gray-800');
  });

  it('renders cancelled status correctly', () => {
    render(<RunStatusBadge status="cancelled" />);
    
    const badge = screen.getByText('cancelled');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-gray-100', 'text-gray-600');
  });

  it('applies additional className', () => {
    render(<RunStatusBadge status="success" className="extra-class" />);
    
    const badge = screen.getByText('success');
    expect(badge).toHaveClass('extra-class');
  });
});
