from setuptools import setup

package_name = 'slz_intellisystem'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='flex',
    maintainer_email='felixfat9@gmail.com',
    description='UAV intelligent information system for landing on an unprepared site',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start = slz_intellisystem.slz_intellisystem:main'
        ],
    },
)
