import matplotlib.pyplot as plt
import io

def render_math(expression, filename):
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.patch.set_alpha(0)
    ax.axis('off')
    ax.text(0.5, 0.5, f"${expression}$", size=14, ha='center', va='center')
    plt.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close(fig)

render_math(r"\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}", "test_math.png")
