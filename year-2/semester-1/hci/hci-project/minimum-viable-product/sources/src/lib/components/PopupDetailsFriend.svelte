<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Avatar from "$lib/components/Avatar.svelte";
    import Popup from "$lib/components/Popup.svelte";
    import { onRemoveFriend } from "$lib/components/PopupDetailsFriend.telefunc";
    import PopupRemoveFriend from "$lib/components/PopupRemoveFriend.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import { HandCoins } from "lucide-svelte";

    let {
        open = $bindable(false),
        user,
        onremovefriend
    }: {
        open?: boolean;
        user: Readonly<{
            id: string;
            displayName: string;
            avatar: boolean;
            avatarModified: Date;
            username: string;
        }>;
        onremovefriend?: () => Promise<void>;
    } = $props();

    let openRemove = $state(false);
    $effect.pre(() => {
        if (!open) openRemove = false;
    });
</script>

<Popup bind:open>
    <div class="flex items-center gap-4">
        <Avatar {user} />
        <p>{user.displayName}</p>
    </div>
    <div class="grid grid-cols-3">
        <p>Username:</p>
        <p class="col-span-2">{user.username}</p>
    </div>
    <hr />
    <Button class="shadow-md" href={"/issueloan?" + new URLSearchParams({ user: user.id })}>
        <AppIcon icon={HandCoins} class="!size-6" />
        Issue Loan
    </Button>
    <PopupRemoveFriend
        bind:open={openRemove}
        {user}
        onconfirm={async () => {
            await onRemoveFriend(user.id);
            await onremovefriend?.();
            openRemove = false;
        }}
    />
</Popup>
