# Replit Deployment Guide

This guide walks you through deploying the SmPC MCP Server on Replit so it can be accessed publicly with username/password authentication.

## Prerequisites

- A Replit account (free tier works)
- Your API keys (Google/OpenAI, etc.)
- Username and password for demo access

## Step-by-Step Deployment

### 1. Create a New Repl

1. Go to [Replit](https://replit.com) and sign in
2. Click **"Create Repl"** or **"+"** button
3. Select **"Import from GitHub"** (if your code is on GitHub) OR **"Python"** template
4. If using GitHub: Enter your repository URL
5. Name your Repl (e.g., `smpc-mcp-server`)

### 2. Upload Your Code

If you didn't import from GitHub:

1. Upload all project files to Replit:
   - Drag and drop files into the Replit file explorer
   - OR use the **"Upload file"** button
   - Ensure all files are uploaded, especially:
     - `run_mcp_server.py`
     - `config.py`
     - `requirements.txt`
     - `src/` directory (all Python files)
     - `web/` directory (HTML, CSS, JS files)
     - `data/` directory (if needed for vector store)

### 3. Configure Environment Variables (Secrets)

**Critical:** Replit uses "Secrets" for environment variables. These are secure and not visible in code.

1. Click the **"Secrets"** tab (🔒 icon) in the left sidebar
2. Add the following secrets (click **"New secret"** for each):

   **Required Secrets:**
   ```
   MCP_AUTH_USERNAME = your_demo_username
   MCP_AUTH_PASSWORD = your_demo_password
   LLM_PROVIDER = gemini  (or gpt5)
   GOOGLE_API_KEY = your_google_api_key  (if using gemini)
   OPENAI_API_KEY = your_openai_api_key  (if using gpt5)
   ```

   **Optional Secrets (for better performance/features):**
   ```
   ENABLE_RERANKING = false  (set to true for better accuracy, slower)
   ENABLE_QUERY_REWRITE = false  (set to true for query enhancement)
   OPIK_API_KEY = your_opik_key  (for observability, optional)
   OPIK_PROJECT_NAME = mcp-poc
   ```

3. Click **"Add secret"** after each entry

**Note:** Replit automatically provides the `PORT` environment variable, which your app already uses.

### 4. Verify .replit Configuration

Your `.replit` file should already be configured correctly:
```
run = "python run_mcp_server.py"
entrypoint = "run_mcp_server.py"
```

If it's missing or incorrect, create/update it:
1. Create a file named `.replit` in the root directory
2. Add the content above

### 5. Install Dependencies

1. Replit should automatically detect `requirements.txt` and install dependencies
2. If not, open the **"Shell"** tab and run:
   ```bash
   pip install -r requirements.txt
   ```
3. Wait for installation to complete (this may take a few minutes)

### 6. Prepare Data Files (If Needed)

If your vector store or data files aren't in the repository:

1. Upload your `data/` directory to Replit
2. Ensure `data/vector_store/` exists (or it will be created on first run)
3. If you have pre-built vector stores, upload them to the correct location

### 7. Run the Application

1. Click the **"Run"** button (▶️) at the top
2. Wait for the server to start (check the console output)
3. You should see: `Application startup complete` and `Uvicorn running on...`

### 8. Make It Publicly Accessible

1. In the Replit interface, look for the **"Webview"** panel on the right
2. Click the **"Open in new tab"** icon (or the URL shown)
3. This opens your app in a new browser tab
4. **Important:** The URL will be something like: `https://your-repl-name.your-username.repl.co`

### 9. Configure Always-On (Optional, Paid Feature)

For free tier:
- Your Repl will sleep after ~1 hour of inactivity
- Visitors will need to wait ~30 seconds for it to wake up
- First request after sleep may be slow (cold start)

For Always-On (Replit Core/Hacker plan):
1. Go to Replit settings
2. Enable **"Always On"**
3. Your app will stay running 24/7

### 10. Test the Deployment

1. Open your public URL in a browser
2. You should see the login form
3. Enter the username and password you set in Secrets
4. After login, you should see the query interface
5. Test with a sample query (e.g., "Aukaverkanir Tegretol")

## Troubleshooting

### Issue: "Module not found" errors
**Solution:** 
- Check that `requirements.txt` is in the root directory
- Run `pip install -r requirements.txt` manually in Shell
- Verify all dependencies are listed in `requirements.txt`

### Issue: "Vector store not initialized"
**Solution:**
- Ensure `data/vector_store/` directory exists
- Check that vector store files are uploaded
- The app will create it on first run if missing, but this may take time

### Issue: "Invalid credentials" on login
**Solution:**
- Verify `MCP_AUTH_USERNAME` and `MCP_AUTH_PASSWORD` are set correctly in Secrets
- Check for typos or extra spaces
- Restart the Repl after changing secrets

### Issue: "API key not found" errors
**Solution:**
- Verify the correct API key secret is set (GOOGLE_API_KEY or OPENAI_API_KEY)
- Check that `LLM_PROVIDER` matches your API key (gemini → GOOGLE_API_KEY)
- Restart the Repl after adding secrets

### Issue: Static files (CSS/JS) not loading
**Solution:**
- Verify `web/` directory is uploaded correctly
- Check that files are named correctly (`style.css`, `app.js`)
- Ensure the server is running (check console output)

### Issue: App sleeps too often
**Solution:**
- Upgrade to Replit Core/Hacker for Always-On
- OR add a simple ping/health check endpoint and use a service like UptimeRobot to ping it every 5 minutes

### Issue: Slow cold starts
**Solution:**
- This is normal for free tier
- The warmup endpoint helps, but first request after sleep will still be slow
- Consider upgrading to Always-On for better performance

## Security Considerations

1. **Never commit `.env` files** - Use Replit Secrets instead
2. **Use strong passwords** - Your `MCP_AUTH_PASSWORD` should be strong
3. **Restrict CORS** - In production, update `allow_origins` in `mcp_server.py` to your specific domain
4. **Monitor usage** - Check Replit logs for suspicious activity

## Sharing Your Demo

Once deployed:

1. Share the public URL: `https://your-repl-name.your-username.repl.co`
2. Provide the username and password separately (via secure channel)
3. Users can access the login page and authenticate

## Updating the Deployment

To update your deployed app:

1. Make changes to your code locally
2. Push to GitHub (if using GitHub import)
3. In Replit, click **"Version control"** → **"Pull latest"**
4. OR manually upload changed files
5. Click **"Run"** to restart with new code

## Cost Considerations

- **Free Tier:** Sleeps after inactivity, ~30s wake time
- **Core ($7/month):** Always-On, better performance
- **Hacker ($20/month):** Always-On, more resources

For demos, free tier works but Always-On provides better UX.

## Next Steps

After successful deployment:

1. Test all features thoroughly
2. Share the URL with demo users
3. Monitor logs for any issues
4. Consider setting up monitoring/alerts

---

**Need Help?** Check the main `README.md` for application-specific details or Replit's documentation for platform-specific issues.
