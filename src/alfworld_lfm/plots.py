import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


large = [
   {
       "lr":       "5e-5",
       "train":    [0.1944, 0.1653, 0.1623, 0.1616, 0.1594, 0.1599, 0.1570,
                    0.1567, 0.1526, 0.1569, 0.1558, 0.1544, 0.1553, 0.1536,
                    0.1541, 0.1531, 0.1530, 0.1526, 0.1525, 0.1520, 0.1517],
       "val":      [0.1607, 0.1621, 0.1588, 0.1548, 0.1526, 0.1532, 0.1520,
                    0.1526, 0.1514, 0.1528, 0.1515],
   },
   {
       "lr":       "3e-6",
       "train":    [0.2604, 0.1991, 0.1946, 0.1920, 0.1907, 0.1876, 0.1863,
                    0.1847, 0.1811, 0.1764, 0.1708, 0.1684, 0.1662, 0.1644,
                    0.1634, 0.1647, 0.1630, 0.1619, 0.1626, 0.1629],
       "val":      [0.1873, 0.1879, 0.1850, 0.1790, 0.1706, 0.1600, 0.1592,
                    0.1582, 0.1574, 0.1569, 0.1629],
   },
   {
       "lr":       "3e-8",
       "train":    [0.7122, 0.6430, 0.5778, 0.5223, 0.4748, 0.4364, 0.4061,
                    0.3846, 0.3692, 0.3569, 0.3470, 0.3395, 0.3332, 0.3290,
                    0.3261, 0.3226, 0.3210, 0.3207, 0.3204, 0.3186],
       "val":      [0.6497, 0.5422, 0.4481, 0.3821, 0.3444, 0.3222, 0.3086,
                    0.3003, 0.2952, 0.2923, 0.2911],
   },
]


base = [
       {
   "lr":       "5e-5",
   "train":    [0.2109, 0.1875, 0.1803, 0.1761, 0.1699, 0.1585, 0.1565, 0.1547,
               0.1553, 0.1535, 0.1535, 0.1536, 0.1527, 0.1524, 0.1529, 0.1523,
               0.1519, 0.1518, 0.1514, 0.1521],
   "val":      [0.1743, 0.1522, 0.1521, 0.1501, 0.1497, 0.1521],
       },
       {
   "lr":       "3e-6",
   "train":    [0.2743, 0.2180, 0.2058, 0.2009, 0.1989, 0.1966, 0.1946, 0.1942,
               0.1919, 0.1921, 0.1920, 0.1907, 0.1904, 0.1888, 0.1882, 0.1885,
               0.1877, 0.1872, 0.1873, 0.1868],
   "val":      [0.1914, 0.1859, 0.1825, 0.1812, 0.1798, 0.1794],
       },
       {
   "lr":       "3e-8",
   "train":    [0.4091, 0.3992, 0.3925, 0.3847, 0.3782, 0.3720, 0.3692, 0.3625,
               0.3576, 0.3554, 0.3520, 0.3484, 0.3478, 0.3452, 0.3441, 0.3429,
               0.3415, 0.3405, 0.3404, 0.3400],
   "val":      [0.3794, 0.3588, 0.3452, 0.3373, 0.3336],
       },
]


small = [
   {
       "lr":       "5e-5",
       "train":    [0.3327, 0.0059, 0.0029, 0.0028, 0.0017, 0.0022, 0.0014, 0.0022,
                   0.0012, 0.0009, 0.0010, 0.0008, 0.0007, 0.0006, 0.0017, 0.0007,
                   0.0009, 0.0010, 0.0005],
       "val":      [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0005],
           },
           {
       "lr":       "3e-6",
       "train":    [2.6562, 0.9874, 0.2767, 0.1091, 0.0619, 0.0418, 0.0338, 0.0277,
                   0.0244, 0.0220, 0.0189, 0.0186, 0.0176, 0.0150, 0.0166, 0.0164,
                   0.0142, 0.0140, 0.0141, 0.0144],
       "val":      [0.0049, 0.0007, 0.0004, 0.0003, 0.0002],
           },
           {
       "lr":       "3e-8",
       "train":    [3.8402, 3.8017, 3.7783, 3.7501, 3.7269, 3.7118, 3.6894, 3.6610,
                   3.6526, 3.6365, 3.6287, 3.6332, 3.6023, 3.5991, 3.5960, 3.5916,
                   3.5834, 3.5768, 3.5753, 3.5757],
       "val":      [3.7594, 3.6826, 3.6254, 3.5919, 3.5755],
           },
]




def create_graphs(model, model_name):
   colors = ["#3A7DCC", "#CC6633", "#2A9E60"]


   fig, axes = plt.subplots(1, 3, figsize=(16, 5))


   fig.suptitle(
       "Training & Validation Loss per Epoch",
       fontsize=20,
       fontweight="bold"
   )


   for ax, job, color in zip(axes, model, colors):
       train = job["train"]
       val   = job["val"]


       train_epochs = np.arange(1, len(train) + 1)
       val_epochs = np.linspace(1, len(train), len(val))


       ax.plot(train_epochs, train,
               color=color, linewidth=2.0, label="Train")


       ax.plot(val_epochs, val,
               color=color, linewidth=2.0, linestyle="--",
               label="Val", alpha=0.75)


       ax.scatter(train_epochs[-1], train[-1],
                  color=color, linewidths=1.2)


       ax.scatter(val_epochs[-1], val[-1],
                  color=color, linewidths=1.2, marker="D")


       ax.annotate(
           f"  {train[-1]:.4f}",
           xy=(train_epochs[-1], train[-1]),
           fontsize=10
       )


       ax.annotate(
           f"  {val[-1]:.4f}",
           xy=(val_epochs[-1], val[-1]),
           fontsize=10,
       )


       ax.set_title(
           f"lr = {job['lr']}",
           fontsize=16,
           fontweight="bold"
       )


       ax.set_xlabel(
           "Epoch",
           fontsize=15
       )


       ax.set_ylabel(
           "Loss",
           fontsize=15
       )


       ax.tick_params(axis='both', labelsize=13)
       ax.grid(True, linestyle="--", alpha=0.5)
       ax.legend(fontsize=12)


   fig.tight_layout(rect=[0, 0, 1, 0.95])


   plt.savefig(
       f"loss_curves_{model_name}.png",
       dpi=300,
       bbox_inches="tight"
   )


   plt.show()


create_graphs(large, "large")
create_graphs(base, "base")
create_graphs(small, "small")