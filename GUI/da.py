t = syp.Symbol("t")

x = syp.Function("x")
y=  syp.Function("y")
z = syp.Function("z")
theta_x = syp.Function("theta_x")
theta_y = syp.Function("theta_y")
theta_z = syp.Function("theta_z")

theta_1 = syp.Function("theta_1")
theta_2 = syp.Function("theta_2")
d_3 = syp.Function("d_3")


joint_variables = syp.Matrix([[syp.diff(theta_1(t))], [syp.diff(theta_2(t))], [syp.diff(d_3(t))]])
position_variables = syp.Matrix([[syp.diff(x(t))], [syp.diff(y(t))], [syp.diff(z(t))], [syp.diff(theta_x(t))], [syp.diff(theta_y(t))], [syp.diff(theta_z(t))]])
syp.init_printing()
print(J)
Jacobian_matrix = syp.Eq(position_variables, J*joint_variables)


print(syp.pretty(Jacobian_matrix))