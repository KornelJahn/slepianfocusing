# TODO: replace by a proper development shell instead of FHS user env

{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSUserEnv {
  name = "slepianfocusing-pipzone";
  targetPkgs = pkgs: (with pkgs; [
    python311
    python311Packages.pip
    zlib
  ]);
  runScript = "bash";
}).env
