import { getGroups } from "$lib/server/db/common_queries";
import { encodeGroup } from "$lib/server/db_utils";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async (event) => {
    await event.parent();
    const user = event.locals.user!;

    return { groups: (await getGroups(user.id)).map(encodeGroup) };
};
