<script module lang="ts">
    export function displayNameValid(displayName: string): boolean {
        return displayName.length > 3;
    }
</script>

<script lang="ts">
    import { invalidateAll } from "$app/navigation";
    import Avatar from "$lib/components/Avatar.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import { Input } from "$lib/shadcn/components/ui/input";
    import { mutation } from "$lib/utils.svelte.js";
    import { encodeBase64urlNoPadding } from "@oslojs/encoding";
    import { onChangeDisplayName, onChangeProfilePic } from "./page.telefunc.js";

    let { data } = $props();

    let profilePicInput: HTMLInputElement | null = $state(null);
    let profilePicFiles: FileList | null = $state(null);

    const mutProfilePic = mutation(async (value: File) => {
        const fileContents = new Uint8Array(await value.arrayBuffer());
        await onChangeProfilePic(encodeBase64urlNoPadding(fileContents));
        await invalidateAll();
    });

    $effect.pre(() => {
        if (profilePicFiles?.length != 1) return;
        mutProfilePic.desiredValue = profilePicFiles[0];
        profilePicFiles = new DataTransfer().files;
    });

    const mutDisplayName = mutation({
        get() {
            return data.user.displayName;
        },
        async set(value) {
            if (value == data.user.displayName) return true;
            if (!displayNameValid(value)) return true;
            await onChangeDisplayName(value);
            await invalidateAll();
        }
    });
</script>

<div class="flex flex-col gap-4 rounded-md border-2 border-neutral-300 p-4">
    <div class="flex items-center">
        <p>Profile picture</p>
        <div class="ml-auto">
            <input
                class="hidden"
                type="file"
                accept="image/png"
                bind:this={profilePicInput}
                bind:files={profilePicFiles}
            />
            <Avatar
                class={["cursor-pointer", mutProfilePic.applying && "animate-pulse"]}
                user={data.user}
                interactionReminder
                onclick={() => profilePicInput?.click()}
            />
        </div>
    </div>
    <div class="flex items-center">
        <p>Display name</p>
        <Input
            class={["ml-auto w-48", mutDisplayName.applying && "animate-pulse"]}
            bind:value={mutDisplayName.value}
        />
    </div>
    <div class="flex items-center">
        <p>Username</p>
        <p class="ml-auto">{data.user.username}</p>
    </div>
</div>

<form class="contents" method="POST" action="/login?/logout">
    <Button class="shadow-md" type="submit">Log Out</Button>
</form>
