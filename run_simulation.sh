#!/bin/bash
echo "Choose a simulator to run:"
echo "1. CARLA"
echo "2. SODA.Sim"
read -p "Enter your choice (1/2): " choice
if [[ $choice -eq 1 ]]; then
    echo "Starting CARLA simulation..."
    docker run -it --rm --name carla-simulation \
        -v $(pwd):/workspace \
        -e "DISPLAY=$DISPLAY" \
        carla-image ./start_carla.sh
elif [[ $choice -eq 2 ]]; then
    echo "Starting SODA.Sim simulation..."
    docker run -it --rm --name soda-simulation \
        -v $(pwd):/workspace \
        -e "DISPLAY=$DISPLAY" \
        soda-image ./start_soda_sim.sh
else
    echo "Invalid choice. Exiting."
    exit 1
fi
