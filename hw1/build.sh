set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip        \
        python3-tk         \
        ;

    ## Unpinned
    # python3 -m pip install -qq             \
    #     matplotlib                         \
    #     numpy                              \
    #     pillow                             \
    #     pytest                             \
    #     scikit-image                       \
    #     scikit-learn                       \
    #     ;

    ## Pinned
    python3 -m pip install -qq             \
        cloudpickle==3.1.2                 \
        contourpy==1.3.3                   \
        cycler==0.12.1                     \
        fonttools==4.64.0                  \
        ImageIO==2.37.4                    \
        iniconfig==2.3.0                   \
        joblib==1.6.0                      \
        kiwisolver==1.5.1                  \
        lazy-loader==0.5                   \
        matplotlib==3.11.1                 \
        narwhals==2.25.0                   \
        networkx==3.6.1                    \
        numpy==2.5.2                       \
        packaging==26.3                    \
        pillow==11.3.0                     \
        pluggy==1.6.0                      \
        Pygments==2.21.0                   \
        pyparsing==3.3.2                   \
        pytest==9.1.1                      \
        python-dateutil==2.9.0.post0       \
        scikit-image==0.26.0               \
        scikit-learn==1.9.0                \
        scipy==1.18.1                      \
        setuptools==84.0.0                 \
        six==1.17.0                        \
        threadpoolctl==3.6.0               \
        tifffile==2026.8.23                \
        wheel==0.42.0                      \
        ;
}

setup_checker() {
    python3 --version # Python 3.12.3
    python3 -m pip freeze # see list above
    python3 -c 'import matplotlib.pyplot'
}

"$@"