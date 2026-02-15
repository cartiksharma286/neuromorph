# How to Install Xcode

To build and simulate your iOS application ("TraderBot_Can"), you **must** install the full version of Xcode. This cannot be done automatically via terminal commands due to its size and Apple ID requirements.

## Option 1: Mac App Store (Recommended)
1.  **Click this link**: [Xcode on the Mac App Store](https://apps.apple.com/us/app/xcode/id497799835?mt=12)
2.  Click **"Get"** or the Cloud icon.
3.  Wait for the installation to complete (it is approx. 3-7 GB).
4.  Once installed, open Xcode from your Applications folder to accept the license agreement.

## Option 2: Apple Developer Website
1.  Go to [developer.apple.com/download/all/](https://developer.apple.com/download/all/)
2.  Sign in with your Apple ID.
3.  Search for "Xcode" and download the latest `.xip` file.
4.  Double-click the `.xip` file to expand it, then drag `Xcode.app` to your `/Applications` folder.

## Post-Installation Setup
After installing, run these commands in your terminal to finish the setup:

```bash
# 1. Select the Xcode directory
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# 2. Agree to the license (so you don't have to click through it)
sudo xcodebuild -license accept

# 3. Verify simulator works
xcrun simctl list
```

## Next Steps
Once Xcode is verified, you can launch your app simulator:

```bash
npx cap open ios
```
