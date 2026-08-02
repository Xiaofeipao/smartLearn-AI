# Day 2 Deployment

## URLs
- Frontend: https://smart-learn-ai-eight.vercel.app
- Backend health: https://smartlearn-ai-production-f127.up.railway.app/health
- Backend docs: https://smartlearn-ai-production-f127.up.railway.app/docs

## Source
- Repository: https://github.com/Xiaofeipao/smartLearn-AI
- Deployed branch: main
- Merged commit: 6884a03
- Pull Request: https://github.com/Xiaofeipao/smartLearn-AI/pull/1

## Root Directories
- Railway: smartlearn-backend
- Vercel: smartlearn-frontend

## Environment variables
- Railway: OPENROUTER_API_KEY, ALLOWED_ORIGINS
- Vercel: VITE_API_URL

## Acceptance results
- /health: pass
- Upload: pass
- Known /chat + citations: pass
- Unknown question: pass
- CORS restart + re-upload: pass

## Known limitations
- Railway restart clears in-memory state; re-upload is expected.