FROM ghcr.io/pangtao22/quasistatic_simulator_base:main

# install
RUN apt-get update
RUN apt-get install vim wget -y
RUN apt-get install make -y
RUN apt-get install openssh-server -y

# git ssh config
RUN mkdir -p /root/.ssh        
RUN chmod -R 600 /root/.ssh/  
COPY setup/id_ed25519 /root/.ssh/id_ed25519
COPY setup/id_ed25519.pub /root/.ssh/id_ed25519.pub
RUN ssh-keyscan github.com >>/root/.ssh/known_hosts

# replace COPY with git clone
# COPY models $QSIM_PATH/models
# COPY robotics_utilities $QSIM_PATH/robotics_utilities
# COPY qsim $QSIM_PATH/qsim
# COPY quasistatic_simulator_cpp $QSIM_CPP_PATH
RUN git clone https://github.com/Director-of-G/quasistatic_simulator.git

# TODO: add new dependencies
# pinocchio
COPY ./setup/install_pinocchio.sh /tmp/
RUN /tmp/install_pinocchio.sh
# mosek
COPY ./setup/install_mosek.sh /tmp/
RUN /tmp/install_mosek.sh
COPY setup/mosek.lic /home/mosek/mosek.lic
ENV MOSEKLM_LICENSE_FILE /home/mosek/mosek.lic

# TODO: build quasistatic simulator
RUN source ~/.bashrc
COPY ./setup/build_bindings.sh /tmp/
RUN /tmp/build_bindings.sh
