import type { ClientInit } from "@sveltejs/kit";
import { config } from "telefunc/client";

export const init: ClientInit = async () => {
    config.telefuncUrl = "/api/_telefunc";
};
