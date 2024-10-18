import datetime        # current date
import math            # trig, trancendentals, etc
import pathlib         # output path
import sys             # floating point constants

# Physical 'constants'

gravity_constant_si_units      = 6.67408E-11
permeability_constant_si_units = 4 * math.pi * 1E-7;

# Unit 'constants'

si_units_to_eotvos = 1E9
si_units_to_nT     = 1E9
gram_per_cubic_cm_to_kg_per_cubic_m = 1E3

# Numeric 'constants'

machine_precision       = sys.float_info.epsilon
sqrt_of_machine_precision = math.sqrt(machine_precision)
numeric_zero            = sqrt_of_machine_precision
small_positive_constant = sqrt_of_machine_precision

# Basis transformations

def spherical_to_cartesian_magnetics(inclination_radians, declination_radians):
	"""spherical_to_cartesian_magnetics transforms magnetic spherical
coordinates to cartesian coordinates in the geophysical convention

Inputs:
inclination (radians) is positive *below* horizontal
declination (radians) is positive E of N

Outputs:
x (northing) unit vector
y (easting)  unit vector
z (positive down) unit vector
"""
	x = math.cos(inclination_radians) * math.cos(declination_radians)
	y = math.cos(inclination_radians) * math.sin(declination_radians)
	z = math.sin(inclination_radians)
	return x, y, z

# Helper functions

def safe_natural_logarithm(x):
	# Small positive constant added for small x
	global small_positive_constant, numeric_zero
	return math.log(x + (small_positive_constant if abs(x) < numeric_zero else 0))

def safe_inverse_tangent(numerator, denominator):
	# Python inhouse math.atan2 follow C style boundary conditions
	# 1. math.atan2(0, 0)        == 0
	# 2. math.atan2(positive, 0) == π / 2
	# 3. math.atan2(negative, 0) == - π / 2
	return math.atan2(numerator, denominator)

# Green's kernel

def partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, x_dummy, y_dummy, z_dummy):

	x_distance = x_eval - x_dummy
	y_distance = y_eval - y_dummy
	z_distance = z_eval - z_dummy
	euclidean_distance = math.hypot(x_distance, y_distance, z_distance)

	return safe_inverse_tangent(y_distance * z_distance, x_distance * euclidean_distance)

def partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, x_dummy, y_dummy, z_dummy):

	x_distance = x_eval - x_dummy
	y_distance = y_eval - y_dummy
	z_distance = z_eval - z_dummy
	euclidean_distance = math.hypot(x_distance, y_distance, z_distance)

	return safe_inverse_tangent(x_distance * z_distance, y_distance * euclidean_distance)

def partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, x_dummy, y_dummy, z_dummy):

	x_distance = x_eval - x_dummy
	y_distance = y_eval - y_dummy
	z_distance = z_eval - z_dummy
	euclidean_distance = math.hypot(x_distance, y_distance, z_distance)

	return safe_inverse_tangent(x_distance * y_distance, z_distance * euclidean_distance)

def partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, x_dummy, y_dummy, z_dummy):

	x_distance = x_eval - x_dummy
	y_distance = y_eval - y_dummy
	z_distance = z_eval - z_dummy
	euclidean_distance = math.hypot(x_distance, y_distance, z_distance)

	return -safe_natural_logarithm(euclidean_distance + z_distance)

def partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, x_dummy, y_dummy, z_dummy):

	x_distance = x_eval - x_dummy
	y_distance = y_eval - y_dummy
	z_distance = z_eval - z_dummy
	euclidean_distance = math.hypot(x_distance, y_distance, z_distance)

	return -safe_natural_logarithm(euclidean_distance + y_distance)

def partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, x_dummy, y_dummy, z_dummy):

	x_distance = x_eval - x_dummy
	y_distance = y_eval - y_dummy
	z_distance = z_eval - z_dummy
	euclidean_distance = math.hypot(x_distance, y_distance, z_distance)

	return -safe_natural_logarithm(euclidean_distance + x_distance)

def greens_kernel_component_xx(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2):

	assert a2 > a1, "a1 must be strictly less than a2"
	assert b2 > b1, "b1 must be strictly less than b2"
	assert c2 > c1, "c1 must be strictly less than c2"

	temp_222 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a2, b2, c2)
	temp_221 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a2, b2, c1)
	temp_212 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a2, b1, c2)
	temp_211 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a2, b1, c1)
	temp_122 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a1, b2, c2)
	temp_121 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a1, b2, c1)
	temp_112 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a1, b1, c2)
	temp_111 = partially_evaluated_greens_kernel_component_xx(x_eval, y_eval, z_eval, a1, b1, c1)

	return temp_222 - temp_221 - temp_212 + temp_211 - temp_122 + temp_121 + temp_112 - temp_111

def greens_kernel_component_yy(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2):

	assert a2 > a1, "a1 must be strictly less than a2"
	assert b2 > b1, "b1 must be strictly less than b2"
	assert c2 > c1, "c1 must be strictly less than c2"

	temp_222 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a2, b2, c2)
	temp_221 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a2, b2, c1)
	temp_212 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a2, b1, c2)
	temp_211 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a2, b1, c1)
	temp_122 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a1, b2, c2)
	temp_121 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a1, b2, c1)
	temp_112 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a1, b1, c2)
	temp_111 = partially_evaluated_greens_kernel_component_yy(x_eval, y_eval, z_eval, a1, b1, c1)

	return temp_222 - temp_221 - temp_212 + temp_211 - temp_122 + temp_121 + temp_112 - temp_111

def greens_kernel_component_zz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2):

	assert a2 > a1, "a1 must be strictly less than a2"
	assert b2 > b1, "b1 must be strictly less than b2"
	assert c2 > c1, "c1 must be strictly less than c2"

	temp_222 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a2, b2, c2)
	temp_221 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a2, b2, c1)
	temp_212 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a2, b1, c2)
	temp_211 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a2, b1, c1)
	temp_122 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a1, b2, c2)
	temp_121 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a1, b2, c1)
	temp_112 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a1, b1, c2)
	temp_111 = partially_evaluated_greens_kernel_component_zz(x_eval, y_eval, z_eval, a1, b1, c1)

	return temp_222 - temp_221 - temp_212 + temp_211 - temp_122 + temp_121 + temp_112 - temp_111

def greens_kernel_component_xy(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2):

	assert a2 > a1, "a1 must be strictly less than a2"
	assert b2 > b1, "b1 must be strictly less than b2"
	assert c2 > c1, "c1 must be strictly less than c2"

	temp_222 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a2, b2, c2)
	temp_221 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a2, b2, c1)
	temp_212 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a2, b1, c2)
	temp_211 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a2, b1, c1)
	temp_122 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a1, b2, c2)
	temp_121 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a1, b2, c1)
	temp_112 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a1, b1, c2)
	temp_111 = partially_evaluated_greens_kernel_component_xy(x_eval, y_eval, z_eval, a1, b1, c1)

	return temp_222 - temp_221 - temp_212 + temp_211 - temp_122 + temp_121 + temp_112 - temp_111

def greens_kernel_component_xz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2):

	assert a2 > a1, "a1 must be strictly less than a2"
	assert b2 > b1, "b1 must be strictly less than b2"
	assert c2 > c1, "c1 must be strictly less than c2"

	temp_222 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a2, b2, c2)
	temp_221 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a2, b2, c1)
	temp_212 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a2, b1, c2)
	temp_211 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a2, b1, c1)
	temp_122 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a1, b2, c2)
	temp_121 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a1, b2, c1)
	temp_112 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a1, b1, c2)
	temp_111 = partially_evaluated_greens_kernel_component_xz(x_eval, y_eval, z_eval, a1, b1, c1)

	return temp_222 - temp_221 - temp_212 + temp_211 - temp_122 + temp_121 + temp_112 - temp_111

def greens_kernel_component_yz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2):

	assert a2 > a1, "a1 must be strictly less than a2"
	assert b2 > b1, "b1 must be strictly less than b2"
	assert c2 > c1, "c1 must be strictly less than c2"

	temp_222 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a2, b2, c2)
	temp_221 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a2, b2, c1)
	temp_212 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a2, b1, c2)
	temp_211 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a2, b1, c1)
	temp_122 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a1, b2, c2)
	temp_121 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a1, b2, c1)
	temp_112 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a1, b1, c2)
	temp_111 = partially_evaluated_greens_kernel_component_yz(x_eval, y_eval, z_eval, a1, b1, c1)

	return temp_222 - temp_221 - temp_212 + temp_211 - temp_122 + temp_121 + temp_112 - temp_111

def gravity_gradient_rectangular_prism_eotvos(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2, density_si_units):
	global gravity_constant_si_units, si_units_to_eotvos

	kernel_component_xx = greens_kernel_component_xx(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_yy = greens_kernel_component_yy(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_zz = greens_kernel_component_zz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_xy = greens_kernel_component_xy(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_xz = greens_kernel_component_xz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_yz = greens_kernel_component_yz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)

	gravity_gradient_component_xx_si_units = gravity_constant_si_units * density_si_units * kernel_component_xx
	gravity_gradient_component_yy_si_units = gravity_constant_si_units * density_si_units * kernel_component_yy
	gravity_gradient_component_zz_si_units = gravity_constant_si_units * density_si_units * kernel_component_zz
	gravity_gradient_component_xy_si_units = gravity_constant_si_units * density_si_units * kernel_component_xy
	gravity_gradient_component_xz_si_units = gravity_constant_si_units * density_si_units * kernel_component_xz
	gravity_gradient_component_yz_si_units = gravity_constant_si_units * density_si_units * kernel_component_yz

	gravity_gradient_component_xx_eotvos = si_units_to_eotvos * gravity_gradient_component_xx_si_units
	gravity_gradient_component_yy_eotvos = si_units_to_eotvos * gravity_gradient_component_yy_si_units
	gravity_gradient_component_zz_eotvos = si_units_to_eotvos * gravity_gradient_component_zz_si_units
	gravity_gradient_component_xy_eotvos = si_units_to_eotvos * gravity_gradient_component_xy_si_units
	gravity_gradient_component_xz_eotvos = si_units_to_eotvos * gravity_gradient_component_xz_si_units
	gravity_gradient_component_yz_eotvos = si_units_to_eotvos * gravity_gradient_component_yz_si_units

	return gravity_gradient_component_xx_eotvos, gravity_gradient_component_yy_eotvos, gravity_gradient_component_zz_eotvos, gravity_gradient_component_xy_eotvos, gravity_gradient_component_xz_eotvos, gravity_gradient_component_yz_eotvos

def total_field_anomaly_rectangular_prism_nT(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2, field_inclination_degrees, field_declination_degrees, magnetization_inclination_degrees, magnetization_declination_degrees, magnetization_si_units):
	global permeability_constant_si_units

	magnetic_scale_factor_si_units = permeability_constant_si_units * magnetization_si_units / (4 * math.pi)

	field_inclination_radians = math.radians(field_inclination_degrees)
	field_declination_radians = math.radians(field_declination_degrees)
	magnetization_inclination_radians = math.radians(magnetization_inclination_degrees)
	magnetization_declination_radians = math.radians(magnetization_declination_degrees)

	field_unit_x, field_unit_y, field_unit_z = spherical_to_cartesian_magnetics(field_inclination_radians, field_declination_radians)
	magnetization_unit_x, magnetization_unit_y, magnetization_unit_z = spherical_to_cartesian_magnetics(magnetization_inclination_radians, magnetization_declination_radians)

	kernel_component_xx = greens_kernel_component_xx(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_yy = greens_kernel_component_yy(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_zz = greens_kernel_component_zz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_xy = greens_kernel_component_xy(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_xz = greens_kernel_component_xz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)
	kernel_component_yz = greens_kernel_component_yz(x_eval, y_eval, z_eval, a1, a2, b1, b2, c1, c2)

	kernel_dot_magnetization_component_x = kernel_component_xx * magnetization_unit_x + \
			kernel_component_xy * magnetization_unit_y + kernel_component_xz * magnetization_unit_z
	kernel_dot_magnetization_component_y = kernel_component_xy * magnetization_unit_x + \
			kernel_component_yy * magnetization_unit_y + kernel_component_yz * magnetization_unit_z
	kernel_dot_magnetization_component_z = kernel_component_xz * magnetization_unit_x + \
			kernel_component_yz * magnetization_unit_y + kernel_component_zz * magnetization_unit_z

	total_field_anomaly_si_units = magnetic_scale_factor_si_units * ( \
			kernel_dot_magnetization_component_x * field_unit_x + \
			kernel_dot_magnetization_component_y * field_unit_y + \
			kernel_dot_magnetization_component_z * field_unit_z)

	total_field_anomaly_nT = si_units_to_nT * total_field_anomaly_si_units

	return total_field_anomaly_nT

if __name__ == "__main__":

	print("Hello from main!")

	# Grid dimensions

	x_minimum =-775
	x_maximum = 800
	x_step = 25

	y_minimum =-775
	y_maximum = 800
	y_step = 25

	z_level = 0

	# Source dimensions

	a1, a2 = -150, 150
	b1, b2 = -150, 150
	c1, c2 =   50, 350

	prism_geometry = (a1, a2, b1, b2, c1, c2)

	# Source physical constants

	magnetization_si_units = 2
	density_gram_per_cubic_cm = 1
	density_kg_per_cubic_m = density_gram_per_cubic_cm * gram_per_cubic_cm_to_kg_per_cubic_m

	field_inclination_degrees = 45
	field_declination_degrees = 20
	magnetization_inclination_degrees = 45
	magnetization_declination_degrees = 20

	# Compute Green's kernel

	x_eval_container = []
	y_eval_container = []
	gravity_gradient_component_xx_eotvos_container = []
	gravity_gradient_component_yy_eotvos_container = []
	gravity_gradient_component_zz_eotvos_container = []
	gravity_gradient_component_xy_eotvos_container = []
	gravity_gradient_component_xz_eotvos_container = []
	gravity_gradient_component_yz_eotvos_container = []

	total_field_anomaly_nT_container = []

	for x_eval in range(x_minimum, x_maximum + x_step, x_step):
		for y_eval in range(y_minimum, y_maximum + y_step, y_step):

			gravity_gradient_component_xx_eotvos, gravity_gradient_component_yy_eotvos, gravity_gradient_component_zz_eotvos, \
			gravity_gradient_component_xy_eotvos, gravity_gradient_component_xz_eotvos, gravity_gradient_component_yz_eotvos = \
					gravity_gradient_rectangular_prism_eotvos(x_eval, y_eval, z_level, \
											   a1, a2, b1, b2, c1, c2, \
											   density_kg_per_cubic_m)

			total_field_anomaly_nT = total_field_anomaly_rectangular_prism_nT(x_eval, y_eval, z_level, \
																	 a1, a2, b1, b2, c1, c2, \
																	 field_inclination_degrees, field_declination_degrees, \
																	 magnetization_inclination_degrees, magnetization_declination_degrees, \
																	 magnetization_si_units)

			x_eval_container.append(x_eval)
			y_eval_container.append(y_eval)

			gravity_gradient_component_xx_eotvos_container.append(gravity_gradient_component_xx_eotvos)
			gravity_gradient_component_yy_eotvos_container.append(gravity_gradient_component_yy_eotvos)
			gravity_gradient_component_zz_eotvos_container.append(gravity_gradient_component_zz_eotvos)
			gravity_gradient_component_xy_eotvos_container.append(gravity_gradient_component_xy_eotvos)
			gravity_gradient_component_xz_eotvos_container.append(gravity_gradient_component_xz_eotvos)
			gravity_gradient_component_yz_eotvos_container.append(gravity_gradient_component_yz_eotvos)

			total_field_anomaly_nT_container.append(total_field_anomaly_nT)

	homework_one_solution_path = pathlib.Path("homework_one_solution.dat")

	last_line = len(x_eval_container)

	with open(homework_one_solution_path, "w") as file:
		metadata = \
f"""# Homework One Solution
#
# x and y in units of meters
# x is Northing (N)
# y is Easting  (E)
# x range is [{x_minimum}, {x_maximum}] inclusive with step size {x_step}
# y range is [{y_minimum}, {y_maximum}] inclusive with step size {y_step}
#
# g_** gravity gradient of component ** due to rectangular prism
# units of g_** in Eötvös
#
# tfa is total-field anomaly due to rectangular prism
# inducing field inclination {field_inclination_degrees} degree (positive below horizontal)
# inducing field declination {field_declination_degrees} degree E of N
# units of tfa in nano Tesla
#
# rectanglular prism geometry in units of meters
# rectangular prism centered at x = {(a2 + a1) / 2}, y = {(b2 + b1) / 2}, z = {(c2 + c1) / 2}
# rectangular prism has x side length of {a2 - a1}
# rectangular prism has y side length of {b2 - b1}
# rectangular prism has z side length of {c2 - c1}
# rectangular prism density contrast is {density_gram_per_cubic_cm} gram per cubic centimeter
# rectangular prism magnetization is {magnetization_si_units} Ampere per meter
# rectangular prism magnetization inclination {magnetization_inclination_degrees} degree (positive below horizontal)
# rectangular prism magnetization declination {magnetization_declination_degrees} degree E of N
#
# Author:
# R Nate Crummett {datetime.date.today()} (robert_crummett@mines.edu)\n"""
		file.write(metadata)

		header = "x y g_xx g_yy g_zz g_xy g_xz g_yz tfa\n"
		file.write(header)

		line_count = 0

		for (x, y, gxx, gyy, gzz, gxy, gxz, gyz, tfa) in zip(
				x_eval_container, y_eval_container, gravity_gradient_component_xx_eotvos_container, \
				gravity_gradient_component_yy_eotvos_container, gravity_gradient_component_zz_eotvos_container, \
				gravity_gradient_component_xy_eotvos_container, gravity_gradient_component_xz_eotvos_container, \
				gravity_gradient_component_yz_eotvos_container, total_field_anomaly_nT_container):
			line_count += 1
			file.write(f"{x} {y} {gxx:.4f} {gyy:.4f} {gzz:.4f} {gxy:.4f} {gxz:.4f} {gyz:.4f} {tfa:.4f}")
			file.write("" if line_count == last_line else "\n")

	print(f"Successfully wrote solution to {homework_one_solution_path}")
