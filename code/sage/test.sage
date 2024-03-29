# Ellipses

var('x_12, x_13, x_14, x_23, x_24, x_34')

eqn = [1490*x_13 + 215*x_34 + 770*x_24 + 215*x_12 + 215*x_23 + 215*x_14 + 0 == 0,
1490*x_13 + 215*x_34 + 770*x_24 + 215*x_12 + 215*x_23 + 215*x_14 + 0 == 0,
1940*x_13 + 230*x_34 + 740*x_24 + 230*x_12 + 230*x_23 + 230*x_14 + 0 == 0,
120*x_13 + 60*x_34 + 2040*x_24 + 60*x_12 + 60*x_23 + 60*x_14 + 0 == 0,
504*x_13 + 108*x_34 + 504*x_24 + 108*x_12 + 108*x_23 + 108*x_14 + 0 == 0,
592*x_13 + 184*x_34 + 1552*x_24 + 184*x_12 + 184*x_23 + 184*x_14 + 0 == 0,
1522*x_13 + 319*x_34 + 1522*x_24 + 319*x_12 + 319*x_23 + 319*x_14 + 0 == 0,
776*x_13 + 92*x_34 + 296*x_24 + 92*x_12 + 92*x_23 + 92*x_14 + 0 == 0,
1774*x_13 + 313*x_34 + 1294*x_24 + 313*x_12 + 313*x_23 + 313*x_14 + 0 == 0,
718*x_13 + 121*x_34 + 478*x_24 + 121*x_12 + 121*x_23 + 121*x_14 + 0 == 0,
770*x_13 + 215*x_34 + 1490*x_24 + 215*x_12 + 215*x_23 + 215*x_14 + 0 == 0,
166*x_13 + 37*x_34 + 166*x_24 + 37*x_12 + 37*x_23 + 37*x_14 + 0 == 0,
338*x_13 + 71*x_34 + 338*x_24 + 71*x_12 + 71*x_23 + 71*x_14 + 0 == 0,
1004*x_13 + 98*x_34 + 284*x_24 + 98*x_12 + 98*x_23 + 98*x_14 + 0 == 0,
754*x_13 + 223*x_34 + 1714*x_24 + 223*x_12 + 223*x_23 + 223*x_14 + 0 == 0,
264*x_13 + 108*x_34 + 1704*x_24 + 108*x_12 + 108*x_23 + 108*x_14 + 0 == 0,
1322*x_13 + 299*x_34 + 1562*x_24 + 299*x_12 + 299*x_23 + 299*x_14 + 0 == 0,
134*x_13 + 53*x_34 + 614*x_24 + 53*x_12 + 53*x_23 + 53*x_14 + 0 == 0,
1224*x_13 + 228*x_34 + 984*x_24 + 228*x_12 + 228*x_23 + 228*x_14 + 0 == 0,
1126*x_13 + 277*x_34 + 1606*x_24 + 277*x_12 + 277*x_23 + 277*x_14 + 0 == 0]

s = solve(eqn, x_12, x_13, x_14, x_23, x_24, x_34);
print(s)

# [
# [x_12 == -r1 - r2 - r3, x_13 == 0, x_14 == r2, x_23 == r3, x_24 == 0, x_34 == r1]
# ]

# -----------------------------------------------------------------
var('x_12, x_13, x_14, x_23, x_24, x_34')

eqn = 1326*x_13 + 57*x_34 + 126*x_24 + 57*x_12 + 57*x_23 + 57*x_14

r1 = 1
r2 = 100
r3 = 99
eqn.subs(x_12 == -r1 - r2 - r3, x_13 == 0, x_14 == r2, x_23 == r3, x_24 == 0, x_34 == r1)

# ------------------------------------------------------------------
var('x_12, x_13, x_14, x_23, x_24, x_34')

eqn = [884*x_13 + 150*x_34 + 420*x_24 + 150*x_12 + 150*x_23 + 150*x_14 + 0 == 0,
254*x_13 + 113*x_34 + 1442*x_24 + 113*x_12 + 113*x_23 + 113*x_14 + 0 == 0,
1326*x_13 + 57*x_34 + 126*x_24 + 57*x_12 + 57*x_23 + 57*x_14 == 0,
510*x_13 + 105*x_34 + 510*x_24 + 105*x_12 + 105*x_23 + 105*x_14 + 0 == 0]

s = solve(eqn, x_12, x_13, x_14, x_23, x_24, x_34);
print(s)

#------------------------
var('x_12, x_13, x_14, x_21, x_23, x_24, x_31, x_32, x_34, x_41, x_42, x_43')

eqn = [61*x_31 + 30*x_34 + 1140*x_24 + 29*x_21 + 31*x_23 + 30*x_14 + 59*x_13 + 31*x_12 + 1140*x_42 + 30*x_43 + 30*x_41 + 29*x_32 + 0 == 0,
761*x_31 + 160*x_34 + 761*x_24 + 159*x_21 + 160*x_23 + 159*x_14 + 761*x_13 + 160*x_12 + 761*x_42 + 159*x_43 + 160*x_41 + 159*x_32 + 0 == 0,
865*x_31 + 168*x_34 + 746*x_24 + 167*x_21 + 167*x_23 + 166*x_14 + 867*x_13 + 167*x_12 + 746*x_42 + 166*x_43 + 168*x_41 + 167*x_32 + 0 == 0,
745*x_31 + 108*x_34 + 385*x_24 + 107*x_21 + 108*x_23 + 107*x_14 + 745*x_13 + 108*x_12 + 385*x_42 + 107*x_43 + 108*x_41 + 107*x_32 + 0 == 0,
679*x_31 + 141*x_34 + 678*x_24 + 140*x_21 + 142*x_23 + 141*x_14 + 677*x_13 + 142*x_12 + 678*x_42 + 141*x_43 + 141*x_41 + 140*x_32 + 0 == 0,
131*x_31 + 55*x_34 + 970*x_24 + 54*x_21 + 56*x_23 + 55*x_14 + 129*x_13 + 56*x_12 + 970*x_42 + 55*x_43 + 55*x_41 + 54*x_32 + 0 == 0,
811*x_31 + 75*x_34 + 211*x_24 + 74*x_21 + 75*x_23 + 74*x_14 + 811*x_13 + 75*x_12 + 211*x_42 + 74*x_43 + 75*x_41 + 74*x_32 + 0 == 0,
563*x_31 + 139*x_34 + 803*x_24 + 138*x_21 + 139*x_23 + 138*x_14 + 563*x_13 + 139*x_12 + 803*x_42 + 138*x_43 + 139*x_41 + 138*x_32 + 0 == 0,
137*x_31 + 52*x_34 + 735*x_24 + 51*x_21 + 54*x_23 + 53*x_14 + 133*x_13 + 54*x_12 + 735*x_42 + 53*x_43 + 52*x_41 + 51*x_32 + 0 == 0,
209*x_31 + 76*x_34 + 811*x_24 + 75*x_21 + 74*x_23 + 73*x_14 + 213*x_13 + 74*x_12 + 811*x_42 + 73*x_43 + 76*x_41 + 75*x_32 + 0 == 0,
635*x_31 + 103*x_34 + 395*x_24 + 102*x_21 + 103*x_23 + 102*x_14 + 635*x_13 + 103*x_12 + 395*x_42 + 102*x_43 + 103*x_41 + 102*x_32 + 0 == 0,
971*x_31 + 55*x_34 + 131*x_24 + 54*x_21 + 55*x_23 + 54*x_14 + 971*x_13 + 55*x_12 + 131*x_42 + 54*x_43 + 55*x_41 + 54*x_32 + 0 == 0,
251*x_31 + 55*x_34 + 252*x_24 + 54*x_21 + 54*x_23 + 53*x_14 + 253*x_13 + 54*x_12 + 252*x_42 + 53*x_43 + 55*x_41 + 54*x_32 + 0 == 0,
853*x_31 + 54*x_34 + 133*x_24 + 53*x_21 + 54*x_23 + 53*x_14 + 853*x_13 + 54*x_12 + 133*x_42 + 53*x_43 + 54*x_41 + 53*x_32 + 0 == 0,
939*x_31 + 131*x_34 + 459*x_24 + 130*x_21 + 131*x_23 + 130*x_14 + 939*x_13 + 131*x_12 + 459*x_42 + 130*x_43 + 131*x_41 + 130*x_32 + 0 == 0,
425*x_31 + 88*x_34 + 424*x_24 + 87*x_21 + 89*x_23 + 88*x_14 + 423*x_13 + 89*x_12 + 424*x_42 + 88*x_43 + 88*x_41 + 87*x_32 + 0 == 0,
273*x_31 + 44*x_34 + 154*x_24 + 43*x_21 + 43*x_23 + 42*x_14 + 275*x_13 + 43*x_12 + 154*x_42 + 42*x_43 + 44*x_41 + 43*x_32 + 0 == 0,
1091*x_31 + 55*x_34 + 131*x_24 + 54*x_21 + 55*x_23 + 54*x_14 + 1091*x_13 + 55*x_12 + 131*x_42 + 54*x_43 + 55*x_41 + 54*x_32 + 0 == 0,
811*x_31 + 75*x_34 + 211*x_24 + 74*x_21 + 75*x_23 + 74*x_14 + 811*x_13 + 75*x_12 + 211*x_42 + 74*x_43 + 75*x_41 + 74*x_32 + 0 == 0,
695*x_31 + 73*x_34 + 216*x_24 + 72*x_21 + 72*x_23 + 71*x_14 + 697*x_13 + 72*x_12 + 216*x_42 + 71*x_43 + 73*x_41 + 72*x_32 + 0 == 0
]

s = solve(eqn, x_12, x_13, x_14, x_21, x_23, x_24, x_31, x_32, x_34, x_41, x_42, x_43);
print(s)

[
[x_12 == -r3 - r4 - r7, x_13 == -r6, x_14 == -r1 + r3 + r4 - 2*r6, x_21 == -r3 - r4 - r5 + 2*r6, x_23 == r7, x_24 == -r2, x_31 == r6, x_32 == r5, x_34 == r4, x_41 == r3, x_42 == r2, x_43 == r1]
]

[
[x_12 == -r3 - r4 - r7, x_13 == -r6, x_14 == -r1 + r3 + r4 - 2*r6, x_21 == -r3 - r4 - r5 + 2*r6, x_23 == r7, x_24 == -r2, x_31 == r6, x_32 == r5, x_34 == r4, x_41 == r3, x_42 == r2, x_43 == r1]
]