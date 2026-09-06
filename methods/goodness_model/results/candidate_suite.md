# Candidate super-chocolate suite

> SYNTHETIC DEMO: members are pool indices, scores are demonstration values. Swap in a real molecule table + real goodness model for a lab-testable suite.

Goodness model held-out R2: 0.6522. Chocolate reference goodness: 0.5679.

| rank | members | predicted | true (oracle) | novelty vs choc | beats choc |
|---|---|---|---|---|---|
| 1 | mol#544, mol#277, mol#354, mol#223 | 0.5306 | 0.6109 | 0.3983 | yes |
| 2 | mol#435, mol#506, mol#130, mol#281 | 0.5303 | 0.5742 | 0.5834 | yes |
| 3 | mol#417, mol#108, mol#130, mol#506 | 0.53 | 0.5842 | 0.9669 | yes |
| 4 | mol#215, mol#33, mol#508, mol#307 | 0.5299 | 0.6158 | 0.6382 | yes |
| 5 | mol#679, mol#214, mol#82, mol#652 | 0.529 | 0.6208 | 0.5562 | yes |
| 6 | mol#631, mol#622, mol#459, mol#586 | 0.5266 | 0.6123 | 0.625 | yes |
| 7 | mol#761, mol#283, mol#56, mol#256 | 0.5257 | 0.5469 | 0.6282 | no |
| 8 | mol#128, mol#28, mol#650, mol#638 | 0.5239 | 0.6129 | 0.4862 | yes |

_Members are the molecules to combine. In a real run each is a named compound with a SMILES and a synthesis/sourcing note for the lab._