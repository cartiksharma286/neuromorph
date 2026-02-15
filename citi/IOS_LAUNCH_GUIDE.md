# TraderBot_Can iOS Store Launch Guide

This guide outlines the steps to package your web application ("TraderBot_Can") as a native iOS app and publish it to the Apple App Store with a price tag of $99.99.

## Phase 1: Prerequisites

1.  **Apple Developer Program Enrollment**
    *   Sign up at [developer.apple.com](https://developer.apple.com/).
    *   Cost: $99/year.
    *   This is required to publish apps and set pricing.

2.  **Environment Setup**
    *   Ensure you have a Mac with Xcode installed (latest version recommended).
    *   Ensure `Node.js` and `npm` are installed (which you clearly have).
    *   Install CocoaPods: `sudo gem install cocoapods`

## Phase 2: Technical Packaging (Using Capacitor)

We will use **Capacitor** to wrap your existing web app into a native iOS container without rewriting code.

1.  **Initialize Capacitor**
    Run the following in your project root:
    ```bash
    npm install @capacitor/core @capacitor/cli @capacitor/ios
    npx cap init "TraderBot_Can" "com.traderbotcan.app"
    ```
    *   *App Name*: TraderBot_Can
    *   *App ID*: com.traderbotcan.app (This must be unique)

2.  **Add iOS Platform**
    ```bash
    npx cap add ios
    ```

3.  **Sync Web Assets**
    Every time you update your `index.html`, `styles.css`, or `app.py` (logic moved to JS), you must sync:
    ```bash
    # Note: Capacitor serves static files. 
    # Since your app uses Flask, you strictly need to decouple the compiled frontend or use a WebView pointing to your server.
    # For a standalone App Store app, it is BEST to convert your Flask logic to client-side JS or host the Flask API remotely.
    
    # If using static frontend:
    npx cap sync
    ```

    *Critical Note*: Apple generally rejects apps that are simple wrappers around a website ("WebView apps") unless they offer "App-like" functionality. Ensure your "Offline Mode" (manifest.json) is robust. Ideally, moving the Flask logic (Simulated Market Data) to pure Client-Side JavaScript avoids the need for a backend server for the app version.

4.  **Open in Xcode**
    ```bash
    npx cap open ios
    ```

## Phase 3: App Store Configuration

1.  **Create App Record**
    *   Log in to **App Store Connect** (appstoreconnect.apple.com).
    *   Click "+" -> "New App".
    *   Select "iOS".
    *   Name: "TraderBot_Can".
    *   SKU: TB_CAN_001.

2.  **Pricing**
    *   Go to **Pricing and Availability**.
    *   Select **Price Schedule**.
    *   Choose **Tier 60** (approx $99.99 USD, check current tier matrix as this changes).
    *   Save.

3.  **Metadata & Assets**
    *   **Screenshots**: You need screenshots for different iPhone sizes (6.5", 5.5", etc.). Use the Simulator in Xcode to take these.
    *   **Description**: "The ultimate AI-powered trading assistant for mineral strategy and market analysis..."
    *   **Keywords**: Trading, AI, Minerals, Dividend, Stocks.

## Phase 4: Build and Submit

1.  **Set App Icon**
    *   In Xcode, go to `AppIcon` in `Assets.xcassets`.
    *   Drag and drop your 1024x1024 icon file.

2.  **Archive**
    *   Select "Any iOS Device (arm64)" as the target.
    *   Menu: **Product** -> **Archive**.
    *   Wait for the build to complete.

3.  **Upload**
    *   Once the Organizer window opens, click **Distribute App**.
    *   Select **App Store Connect**.
    *   Follow the validation prompts.

4.  **Submit for Review**
    *   Go back to App Store Connect.
    *   You should see your build appear under "Builds".
    *   Select it, and click **Submit for Review**.

## Compliance Checklist

*   [ ] **Review Guidelines**: Ensure you don't mention "Android" or other platforms.
*   [ ] **Privacy Policy**: You need a URL to a privacy policy (you can host a simple static page).
*   [ ] **In-App Purchases**: Since this is a paid app ($99.99), you don't need IAP necessarily, but ensure no hidden unlockables violate guidelines.
