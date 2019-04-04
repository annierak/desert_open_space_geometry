import os

for i in range(50):
    os.rename(
    'trap_time_course_by_wind_wind_mag_1.4_wind_angle_2.74_iter_'
    +str(i)+'.pkl',
    'trap_time_course_by_wind_wind_mag_1.4_wind_angle_2.74_iter_'
    +str(i+50)+'.pkl',
    )
