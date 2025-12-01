// See https://svelte.dev/docs/kit/types#app.d.ts

import type { SessionValidationResult } from "$lib/server/auth";

// for information about these interfaces
declare global {
    namespace App {
        // interface Error {}
        interface Locals {
            user: SessionValidationResult["user"];
            session: SessionValidationResult["session"];
        }
        // interface PageData {}
        // interface PageState {}
        // interface Platform {}
    }
}
