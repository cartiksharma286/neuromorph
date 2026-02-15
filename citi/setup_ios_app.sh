#!/bin/bash

# TraderBot_Can iOS Project Setup Script
echo "🍎 Setting up iOS Project for TraderBot_Can..."

# 1. Check/Install CocoaPods (Required for Capacitor iOS)
if ! command -v pod &> /dev/null
then
    echo "❌ CocoaPods not found. Attempting to install via Homebrew..."
    # Attempt to use brew if gem failed previously
    brew install cocoapods || {
        echo "⚠️  Homebrew install failed. Trying Gem (requires user permission)..."
        echo "   Please run: 'sudo gem install cocoapods' manually if this script fails."
        # We won't run sudo here to avoid hanging the agent
        exit 1
    }
fi

# 2. Add iOS Platform
if [ ! -d "ios" ]; then
    echo "📱 Adding iOS platform..."
    npx cap add ios
else
    echo "✅ iOS platform already exists."
fi

# 3. Sync Web Assets
echo "🔄 Syncing web assets..."
npx cap sync ios

# 4. Open Xcode
echo "🚀 Opening Xcode..."
npx cap open ios

echo "✅ Setup Complete. In Xcode:"
echo "   1. Select your simulator (e.g., iPhone 15 Pro)."
echo "   2. Press Cmd+R to run."
