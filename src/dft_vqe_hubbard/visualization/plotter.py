import os

import matplotlib.pyplot as plt


class ResultPlotter:
    """
    Handles the generation and saving of physics result plots.

    Encapsulates file I/O logic, ensuring directories exist and consistent
    styling is applied across different analysis scripts.
    """

    def __init__(self, output_dir: str = "plots", dpi: int = 300) -> None:
        """
        Args:
            output_dir: The directory where plots will be saved.
                        Will be created if it doesn't exist.
            dpi: Resolution for saved images. Defaults to 300 (Publication Quality).
        """
        self._output_dir = output_dir
        self._dpi = dpi

        os.makedirs(self._output_dir, exist_ok=True)

    def plot_mott_transition(
        self,
        u_values: list[float],
        energies: list[float],
        double_occupancies: list[float],
        filename: str = "mott_transition.png",
        title: str = "Mott Transition",
    ) -> None:
        """
        Generates the dual-axis plot for Energy and Double Occupancy.

        Args:
            u_values: X-axis values (Interaction Strength U/t).
            energies: Y1-axis values (Ground State Energy).
            double_occupancies: Y2-axis values (Double Occupancy).
            filename: Name of the output file.
            title: Chart title.
        """
        _, ax1 = plt.subplots(figsize=(10, 6))

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

        self._save_current_figure(filename)

    def _save_current_figure(self, filename: str) -> None:
        """Helper to save and close the current matplotlib figure."""
        full_path = os.path.join(self._output_dir, filename)
        plt.tight_layout()
        plt.savefig(full_path, dpi=self._dpi)
        plt.close()
        print(f"Plot saved to: {full_path}")
