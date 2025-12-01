import type { SessionValidationResult } from "$lib/server/auth";
import "telefunc";

declare module "telefunc" {
    namespace Telefunc {
        interface Context {
            user: SessionValidationResult["user"];
            session: SessionValidationResult["session"];
            /* Globally define the type of the `context` object here, see https://telefunc.com/getContext#typescript
			 * For example:
			user: null | { id: number, name: string }
			*/
        }
    }
}
