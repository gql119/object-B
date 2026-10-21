# Phase B complete robustness report

All formal values are fractions. last.pt@200 is primary; best.pt is secondary. Clean validation is the unchanged 4,952-image split. AT NOT_RUN.

| Defense | Clean Person AP50 | S4 Person AP50 | Clean NT AP50 | S4 NT AP50 | P | U | S | WCD | TCD3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gauss | 0.851175346919 | 0.095012808739 | 0.747003404439 | 0.678606918833 | 0.888374576304 | 0.908438856904 | 0.898294691797 | 0.323968407347 | 0.240520415366 |

## gauss
Clean/S4 Person Recall: 0.736545881581/0.00927561837456; NT Recall: 0.671716894748/0.619381230379.
S4 NT AP75=0.496167199472; NT AP50–95=0.465856245842; all-class AP50=0.649427213328.
Clean checkpoint=82e9857a4013cb095ca1ca29afd1d08fc0153c8ae32ebbffe6225dedac7aff97; S4 checkpoint=772f543908763b61194bb6e18e6fb8ecc898424c4b86c854488dc38d4aa595cd.
Clean control: verified historical200, reevaluated.
| meanf | 0.849968080663 | 0.0692327108508 | 0.740200300484 | 0.660637906904 | 0.918546693193 | 0.892512346283 | 0.905342395187 | 0.312484884192 | 0.258180270905 |

## meanf
Clean/S4 Person Recall: 0.732318550028/0.011887423858; NT Recall: 0.668585353722/0.609648962248.
S4 NT AP75=0.478818068195; NT AP50–95=0.451655234146; all-class AP50=0.631067647102.
Clean checkpoint=edae8db06498bdca7b55726db41a0868c5901f558f662d036f85a49dbe6cb223; S4 checkpoint=121bfcda7f43b30749488a94349c57fa26dfdfc2e73ea2f3a58842a9bba6e409.
Clean control: verified historical200, reevaluated.
| medianf | 0.851928948229 | 0.0936543706927 | 0.742060983281 | 0.666146742293 | 0.890067862012 | 0.897698110131 | 0.893866702988 | 0.349323089979 | 0.297976934025 |

## medianf
Clean/S4 Person Recall: 0.738736749117/0.0152385159011; NT Recall: 0.677594013095/0.612452668023.
S4 NT AP75=0.48461014252; NT AP50–95=0.456294486758; all-class AP50=0.637522123713.
Clean checkpoint=68bb89dc8223ca009a2a78ca47d5bcb9b00d9a2bdb75a3174c21d47f70435a66; S4 checkpoint=9baca1bd6677dec5fd13d2264ab3a70beb147e6b998de2926764230accb91b33.
Clean control: verified historical200, reevaluated.
| data_aug | 0.867713797828 | 0.0337932762757 | 0.768432081581 | 0.688691005992 | 0.961054812819 | 0.896228856784 | 0.927510503965 | 0.289942905517 | 0.25475438028 |

## data_aug
Clean/S4 Person Recall: 0.77401500343/0.00069129984204; NT Recall: 0.70857478476/0.626373533796.
S4 NT AP75=0.499707872334; NT AP50–95=0.463930595886; all-class AP50=0.655946119506.
Clean checkpoint=fb7ed3eea7ed68ac8412132362586b0c0403386e01152a2f51a86dd53628e341; S4 checkpoint=cba7c54187296e32770d58af0494987c0942a62826a882ae41ef1297c181a504.
Clean control: verified historical200, reevaluated.

Total recorded pair wall time: 61940.911 seconds (historical control training cost excluded).
Complete AP50/AP75/AP50–95/Recall/Precision for VOC20 are in the accompanying per-class CSV. High S alone does not establish overall superiority. These are one-seed descriptive comparisons, without a statistical significance or causal mechanism claim.
