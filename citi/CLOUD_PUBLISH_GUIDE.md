# How to Publish without Local Xcode

Since you cannot use Xcode locally, you must use a **Cloud Build Service**. The most compatible service for this project (built with Capacitor) is **Ionic Appflow**.

## Step 1: Create an Ionic Appflow Account
1.  Go to [ionic.io/appflow](https://ionic.io/appflow).
2.  Sign up for an account (Free trial available, but "Native Builds" often require a paid tier).

## Step 2: Push Your Code to Git
You need to host your code on GitHub, GitLab, or Bitbucket so Appflow can see it.
1.  Initialize Git in your folder (if not done):
    ```bash
    git init
    git add .
    git commit -m "Ready for deployment"
    ```
2.  Create a repo on GitHub and push your code there.

## Step 3: Link to Appflow
1.  In the Ionic Dashboard, click **New App**.
2.  Select "Import app from GitHub".
3.  Choose your repository.

## Step 4: Create a Native Build (iOS)
1.  Go to the **Build** tab in Appflow.
2.  Select **iOS**.
3.  **Build Type**: Select "App Store" (Release).
4.  **Signing**: You will need to upload your Apple Developer Certificate (.p12) and Provisioning Profile. 
    *   *Note: You still need an Apple Developer Account ($99/year) to generate these certificates, even if you don't have Xcode.*

## Step 5: Deploy
Once the build finishes on their servers:
1.  Appflow will give you an **.ipa file**.
2.  You can use **Transporter app** (available on the Mac App Store) to upload that .ipa file to App Store Connect.
    *   *Alternatively, Appflow can upload it automatically if you configure the integration.*

## Free Alternative: Progressive Web App (PWA)
If you don't want to pay for cloud builds or Apple Developer fees ($99), you can "publish" immediately as a PWA:
1.  Host your `www` folder on any web host (Vercel, Netlify, GitHub Pages).
2.  Users navigate to the URL on their iPhone.
3.  User taps **Share -> Add to Home Screen**.
4.  The app installs and behaves exactly like a native app (full screen, offline support).
