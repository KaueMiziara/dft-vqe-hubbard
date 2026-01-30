import matplotlib.pyplot as plt


def plot_mott_transition(
    u_values: list[float],
    energies: list[float],
    double_occupancies: list[float],
    save_path: str,
    title: str = "Mott Transition",
) -> None:
    """Generates and saves the dual-axis plot for the Mott Transition.

    Args:
        u_values: List of Interaction Strength (U/t) values.
        energies: List of corresponding Ground State Energies.
        double_occupancies: List of corresponding Double Occupancy expectations.
        save_path: File path to save the generated image.
        title: Title of the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_e = "tab:blue"
    ax1.set_xlabel("Interaction Strength (U/t)")
    ax1.set_ylabel("Ground State Energy ($E_0$)", color=color_e)
    ax1.plot(u_values, energies, color=color_e, marker="o", label="$E_0$")
    ax1.tick_params(axis="y", labelcolor=color_e)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_d = "tab:red"
    ax2.set_ylabel("Double Occupancy ($\\langle D \\rangle$)", color=color_d)
    ax2.plot(
        u_values,
        double_occupancies,
        color=color_d,
        marker="s",
        linestyle="--",
        label="Double Occ.",
    )
    ax2.tick_params(axis="y", labelcolor=color_d)

    plt.title(title)
    fig.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {save_path}")
