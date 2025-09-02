# Thinkerbell Frontend

A React TypeScript frontend for the Thinkerbell DAWN pipeline management system. This admin interface allows you to monitor pipeline runs, manage artifacts, and trigger jobs.

## Features

- **Dashboard**: System health overview with quick actions and recent activity
- **Pipeline Runs**: Monitor and manage pipeline executions with real-time status
- **Artifacts**: Browse, preview, and download generated artifacts (models, reports, zines)
- **Settings**: Configure API endpoints and interface preferences
- **Real-time Updates**: Auto-refreshing data for live pipeline monitoring
- **Mock Mode**: Built-in MSW mocks for development without a backend

## Tech Stack

- **React 18** + **TypeScript** with Vite
- **TailwindCSS** for styling
- **TanStack Query** for data fetching and caching
- **Zod** for runtime validation
- **React Router v6** for routing
- **MSW** for API mocking during development
- **Vitest** + Testing Library for testing

## Quick Start

### Installation

```bash
npm install
```

### Development

```bash
# Start development server (uses MSW mocks by default)
npm run dev
```

The app will start at `http://localhost:5173` with mock data.

### Production Build

```bash
npm run build
npm run preview
```

### Testing

```bash
# Run tests
npm test

# Run tests once
npm run test:run

# Run tests with UI
npm run test:ui
```

## Configuration

### API Connection

By default, the app uses MSW mocks for development. To connect to a real API:

1. Set the `VITE_API_URL` environment variable:
   ```bash
   export VITE_API_URL=https://api.thinkerbell.local/v1
   npm run dev
   ```

2. Or configure it in the Settings page within the app

### Environment Variables

- `VITE_API_URL`: API base URL (defaults to mocks if not set)

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── DataTable.tsx    # Generic table component
│   ├── HealthPill.tsx   # System health indicator
│   ├── Layout.tsx       # Main app layout
│   ├── MetricCard.tsx   # Metric display card
│   ├── RunStatusBadge.tsx # Status badge component
│   └── StartJobButton.tsx # Job trigger button
├── lib/
│   ├── api.ts          # API client with Zod validation
│   ├── schemas.ts      # Zod schemas for type safety
│   └── utils.ts        # Utility functions
├── mocks/
│   ├── browser.ts      # MSW browser setup
│   └── handlers.ts     # API mock handlers
├── pages/              # Page components
│   ├── ArtifactDetail.tsx
│   ├── ArtifactsList.tsx
│   ├── Dashboard.tsx
│   ├── RunDetail.tsx
│   ├── RunsList.tsx
│   └── Settings.tsx
└── test/
    └── setup.ts        # Test configuration
```

## API Contract

The frontend expects these endpoints:

- `GET /runs` - List pipeline runs
- `GET /runs/:id` - Get run details
- `POST /runs` - Start new run
- `GET /artifacts` - List artifacts
- `GET /artifacts/:id` - Get artifact details
- `POST /artifacts/report` - Generate report
- `GET /health` - System health status

See `src/lib/schemas.ts` for complete type definitions.

## Keyboard Shortcuts

- `g d` - Go to Dashboard
- `g r` - Go to Runs
- `g a` - Go to Artifacts

*Note: Keyboard shortcuts are displayed but not yet implemented*

## Development Notes

### Mock Data

The app includes realistic mock data via MSW for development:
- 4 sample pipeline runs with different statuses
- 4 sample artifacts of various types
- Health status simulation

### Type Safety

All API responses are validated at runtime using Zod schemas, ensuring type safety even with external data.

### Styling

Uses TailwindCSS with custom color scheme:
- Success: Emerald tones
- Warning/Running: Amber tones  
- Error/Failed: Rose tones

### Testing

- Unit tests for API client and key components
- MSW for mocking in tests
- React Testing Library for component tests

## Contributing

1. Follow the existing code style (functional components, hooks)
2. Keep components pure (no API calls in presentational components)
3. Use React Query for all data fetching
4. Validate all API responses with Zod
5. Write tests for new functionality
6. Ensure responsive design

## Troubleshooting

### MSW Issues

If you see MSW-related errors:
1. Ensure `public/mockServiceWorker.js` exists
2. Check browser console for MSW registration logs
3. Clear browser cache and reload

### Build Issues

If TypeScript compilation fails:
1. Check for any missing dependencies
2. Ensure all imports are correctly typed
3. Run `npm run build` to see detailed errors

## License

Internal use only - Thinkerbell team.