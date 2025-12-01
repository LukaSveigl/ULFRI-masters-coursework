<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import HandFinger from "$lib/components/icons/PointerFilled.svelte";
    import type { OverrideSvelteHTMLElement } from "$lib/utils.svelte";
    import { sha256 } from "@oslojs/crypto/sha2";
    import { decodeBase64urlIgnorePadding } from "@oslojs/encoding";

    let {
        user,
        interactionReminder,
        isGroup,
        size = "normal",
        class: klass,
        ...props
    }: OverrideSvelteHTMLElement<
        "div",
        {
            user: Readonly<
                {
                    id: string | null;
                    displayName: string;
                } & (
                    | {
                          avatar: true;
                          avatarModified: Date;
                      }
                    | {
                          avatar: false | string;
                      }
                )
            > | null;
            interactionReminder?: boolean;
            isGroup?: boolean;
            size?: "normal" | "small";
        }
    > = $props();

    const reFullname = /^(\w)\w*\s+(\w).*$/;
    const reUsername = /^(\w\w?).*$/;

    function idToColor(id: string): string {
        const idHash = new DataView(sha256(decodeBase64urlIgnorePadding(id)).buffer);
        const hue = (idHash.getUint16(0, true) / 65535) * 360;
        return `${hue.toFixed(0)} 80% 80%`;
    }

    function displayNameToInitials(displayName: string): string {
        const matchFullname = displayName.match(reFullname);
        if (matchFullname != null) {
            return (matchFullname[1] + matchFullname[2]).toUpperCase();
        }

        const matchUsername = displayName.match(reUsername);
        if (matchUsername != null) {
            return matchUsername[1].toUpperCase();
        }

        return "";
    }
</script>

<div
    style:--id-color={user?.id != null ? idToColor(user.id) : "0 0% 100%"}
    {...props}
    class={[
        "relative grid flex-none rounded-full",
        size == "normal" && "size-12",
        size == "small" && "size-6",
        klass
    ]}
>
    <div
        class="grid overflow-hidden rounded-full border-2 border-current bg-[hsl(var(--id-color))] shadow-md"
    >
        {#if user?.avatar}
            <img
                class="pointer-events-none size-full object-cover"
                src={user.avatar === true
                    ? `/api/${isGroup ? "group" : "profile"}picture/${user.id}?m=${user.avatarModified.valueOf()}`
                    : user.avatar}
                alt={displayNameToInitials(user.displayName)}
            />
        {:else}
            <div
                class={[
                    "pointer-events-none flex select-none items-center justify-center font-bold leading-none text-black",
                    size == "normal" && "text-xl",
                    size == "small" && "text-[0.6rem]"
                ]}
            >
                {user != null ? displayNameToInitials(user.displayName) : "?"}
            </div>
        {/if}
    </div>
    {#if interactionReminder}
        <AppIcon
            class="absolute bottom-0 right-0 size-4 drop-shadow-md"
            fill="#eeeeee"
            icon={HandFinger}
        />
    {/if}
</div>
