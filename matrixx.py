Ac = ca.MX.zeros(12, 12)

# First Row
Ac[0, 3] = 1

# Second row
Ac[1, 4] = 1

# third row
Ac[2, 5] = 1

# forth row
Ac[3, 7] = self.g

# fifth row
Ac[4, 6] = -self.g

# 6:th row = Empty


# 7:th row

Ac[6, 6] = w_z
Ac[6, 9] = 1

# 8:th row
Ac[7, 7] = -w_z
Ac[7, 10] = 1

# 9:th row
Ac[8, 8] = w_y
Ac[8, 11] = 1

# 10:th row

Ac[9, 10] = (-w_z * M_z + w_z * M_y) / M_x
Ac[9, 11] = (-w_y * M_z + w_y * M_y) / M_x

# 11:th row

Ac[10, 9] = (w_z * M_z - w_z * M_x) / M_y
Ac[10, 11] = (-w_x * M_z + w_x * M_x) / M_y

# 12:th row

Ac[11, 9] = (-w_y * M_y + w_y * M_x) / M_z
Ac[11, 10] = (-w_x * M_y + w_x * M_x) / M_z

print(Ac)

Bc = ca.MX.zeros(12, 1)

Bc[5, 0] = 1 / self.m
Bc[9, 0] = 1 / M_x
Bc[10, 0] = 1 / M_y
Bc[11, 0] = 1 / M_z
