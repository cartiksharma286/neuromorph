# How to Build iOS App with GitHub Actions (Free)

Since you don't want to use Appflow and don't have local Xcode, the best industry-standard solution is **GitHub Actions**. GitHub provides free macOS virtual machines that *do* have Xcode installed.

## Step 1: Push code to GitHub
1.  Create a new repository on GitHub.com.
2.  Run these commands in your *terminal* (inside the `citi` folder):

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## Step 2: Trigger the Build
1.  Once pushed, go to the **Actions** tab in your repository.
2.  You will see "Build iOS App" running.
3.  Wait for it to finish (approx. 5-10 mins).

## Step 3: Download the App
1.  Click on the completed run.
2.  Scroll down to **Artifacts**.
3.  Download **ios-build**.
4.  Inside is your `App.ipa`.

## Step 4: Sign and Submit
The generated `.ipa` is unsigned. To submit to the App Store:
1.  You (or a friend with a Mac) need to sign it using **Apple Transporter** or a signing tool.
2.  *Advanced*: You can configure the GitHub Action to sign it automatically if you upload your Apple Certificates to GitHub Secrets.
