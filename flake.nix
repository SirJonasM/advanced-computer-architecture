{
  # ... inputs stay the same
  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

	  sync-project = pkgs.writeShellScriptBin "sync-project" ''
        # 1. Capture the current SSID
        # We filter for the active connection that is a wifi type
		WIFI=$(${pkgs.networkmanager}/bin/nmcli device wifi show | grep '^SSID:' | cut -d ':' -f2 | xargs)

        if [ "$WIFI" != "robot2" ]; then
          echo "📡 Switching from $PREVIOUS_WIFI to robot2..."
          ${pkgs.networkmanager}/bin/nmcli device wifi connect robot13
        else
          echo "📶 Already connected to robot2."
        fi

        # Direnv sets the DIRENV_DIR variable. 
        # We strip the leading '-' to get the actual path.
        PROJECT_ROOT="''${DIRENV_DIR#-}"

        # Fallback: if script is run outside direnv, use git
        if [ -z "$PROJECT_ROOT" ]; then
          PROJECT_ROOT=$( ${pkgs.git}/bin/git rev-parse --show-toplevel 2>/dev/null || pwd )
        fi

        echo "🚀 Syncing from $PROJECT_ROOT to Robot ..."
        
        # Perform rsync using the absolute path of the project root
        ${pkgs.rsync}/bin/rsync -azP \
          --include "*.py" \
          --include "*.h5" \
          "$PROJECT_ROOT/code/" "robot:/home/student/code/"

        echo "✅ Sync complete."

		ssh robot
      '';

    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          rsync
		  sync-project
		  typst
		  (python3.withPackages (ps: with ps; [
			  pip
			  pandas
			  matplotlib
			  numpy
			  torch 
			  rpi-gpio
			  torchvision
		  ]))
        ];

        shellHook = ''
          echo "--- ACA Development Shell ---"
        '';
      };
    };
}
