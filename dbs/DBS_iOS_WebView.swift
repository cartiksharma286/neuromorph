import SwiftUI
import WebKit

struct WebView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        return WKWebView()
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        let request = URLRequest(url: url)
        webView.load(request)
    }
}

@main
struct DBSApp: App {
    var body: some Scene {
        WindowGroup {
            // Replace this with the dynamically assigned IP if hosting remotely
            WebView(url: URL(string: "http://192.168.2.14:5006")!)
                .ignoresSafeArea()
        }
    }
}
