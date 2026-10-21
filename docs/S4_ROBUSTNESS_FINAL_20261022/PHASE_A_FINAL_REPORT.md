# Phase A complete robustness report

All formal values are fractions. last.pt@200 is primary; best.pt is secondary. Clean validation is the unchanged 4,952-image split. AT NOT_RUN.

| Defense | Clean Person AP50 | S4 Person AP50 | Clean NT AP50 | S4 NT AP50 | P | U | S | WCD | TCD3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| jpeg50 | 0.865868647428 | 0.0721992850432 | 0.763383336389 | 0.67912331675 | 0.916616353696 | 0.889622925176 | 0.90291793715 | 0.290213226313 | 0.241088406551 |

## jpeg50
Clean/S4 Person Recall: 0.772084805654/0.0059628975265; NT Recall: 0.700947913924/0.614551516589.
S4 NT AP75=0.499142679771; NT AP50–95=0.467952744952; all-class AP50=0.648777115164.
Clean checkpoint=cf9468d667cdfa3015fc9e928ee530e341a662ae8b42eb369f4cbdbd1e6bf12b; S4 checkpoint=971278bc2cf0303e56bfd6faf3329cee1d0092a99991d7a219467731f4d8cd6b.
Clean control: fresh paired clean200.
| jpeg30 | 0.855396434492 | 0.21612030739 | 0.757089800318 | 0.684108512186 | 0.747344858273 | 0.903602864414 | 0.818079149764 | 0.287642901903 | 0.226041941749 |

## jpeg30
Clean/S4 Person Recall: 0.74413200529/0.0398512240262; NT Recall: 0.673685981226/0.629320829325.
S4 NT AP75=0.501869370371; NT AP50–95=0.471973604405; all-class AP50=0.660709101946.
Clean checkpoint=958e805b577594afa92f645791e076e73647148375f3e6ca309a71e27396a4cc; S4 checkpoint=a15462adf57bfee8701163023d8c50647ad24c673b2f0538289786d888a2ec9a.
Clean control: fresh paired clean200.
| gray | 0.850729236916 | 0.0158094661326 | 0.759428728699 | 0.678860858602 | 0.981416571282 | 0.89390989957 | 0.935621612884 | 0.320781219641 | 0.257043843988 |

## gray
Clean/S4 Person Recall: 0.738806757996/0.00132508833922; NT Recall: 0.691762613967/0.609758260505.
S4 NT AP75=0.498032534163; NT AP50–95=0.467927370673; all-class AP50=0.645708288979.
Clean checkpoint=b8247a7e5820b4f37d98a558808fe9f1f264016528461cf0ef39d8591a769390; S4 checkpoint=62007a2294e278aa6d3f94e8d308d206c28b0de638bf41a518cd76e5c81b47c9.
Clean control: fresh paired clean200.
| bdr | 0.868471160296 | 0.0283812032104 | 0.766823981016 | 0.695745784053 | 0.967320500083 | 0.907308327957 | 0.936353834318 | 0.263181832812 | 0.236967470477 |

## bdr
Clean/S4 Person Recall: 0.771234453942/0.000441696113074; NT Recall: 0.705410632071/0.641446433991.
S4 NT AP75=0.51341678565; NT AP50–95=0.483715099696; all-class AP50=0.662377555011.
Clean checkpoint=46e7c9734ea28f57309c27dd0aada1f796a1e916db96798ebde25c4f0829351c; S4 checkpoint=78d1172b6c2fb0d8c0460da5d4e838fc173d78beb81efeca5eef828b6d18b82b.
Clean control: fresh paired clean200.

Total recorded pair wall time: 123402.638 seconds (historical control training cost excluded).
Complete AP50/AP75/AP50–95/Recall/Precision for VOC20 are in the accompanying per-class CSV. High S alone does not establish overall superiority. These are one-seed descriptive comparisons, without a statistical significance or causal mechanism claim.
