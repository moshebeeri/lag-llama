# Creating a Private Fork of Lag-Llama

## Step 1: Fork and Make Private
1. Go to https://github.com/time-series-foundation-models/lag-llama
2. Click the "Fork" button in the top right
3. After forking, go to your fork's settings
4. Scroll down to "Danger Zone"
5. Click "Change repository visibility"
6. Select "Make private"
7. Confirm the change

## Step 2: Clone Your Private Fork
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/lag-llama.git lag-llama-private
cd lag-llama-private

# Add upstream to track the original repository
git remote add upstream https://github.com/time-series-foundation-models/lag-llama.git

# Verify remotes
git remote -v
```

## Step 3: Add Your Development Files
```bash
# Create necessary directories
mkdir -p data models notebooks/{development,production} src/{data,models,utils}

# Copy your development files
cp tech_forecast.py src/models/
cp tech_requirements*.txt .
cp config.yaml .
cp setup_environments.md .

# Add new files to git
git add .
git commit -m "Add development environment and analysis tools"
git push origin main
```

## Step 4: Protect Your Private Repository
1. Go to your fork's settings on GitHub
2. Under "Security" section:
   - Enable "Require pull request reviews before merging"
   - Enable "Require status checks to pass before merging"
   - Enable "Include administrators"

## Step 5: Sync with Upstream (Original Repository)
To keep your private fork updated with the original repository:
```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream changes to your main branch
git checkout main
git merge upstream/main

# Push to your private repository
git push origin main
```

## Step 6: Development Workflow
1. Create feature branches from main:
```bash
git checkout -b feature/new-analysis
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add new analysis feature"
```

3. Push to your private repository:
```bash
git push origin feature/new-analysis
```

4. Create a pull request in your private repository to merge the feature branch into main

## Important Notes:
1. Never push sensitive data (API keys, credentials)
2. Use `.gitignore` to exclude data files and model weights
3. Regularly sync with upstream for important updates
4. Keep your private fork accessible only to trusted team members
5. The fork will maintain the original repository's history
6. You can still receive updates from the original repository through the upstream remote 